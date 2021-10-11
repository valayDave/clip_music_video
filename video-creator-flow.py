from botocore.validate import validate_parameters
from metaflow import FlowSpec,step,batch,Parameter,IncludeFile,retry,resources
import click
import math

sideX       = 512
sideY       = 512

def sigmoid(x):
    x = x * 2. - 1.
    return math.tanh(1.5*x/(math.sqrt(1.- math.pow(x, 2.)) + 1e-6)) / 2 + .5


class VideoGenerationPipeline(FlowSpec):

    batch_size = Parameter('batch-size',default = 64,type=int,help='Batch size to use for training the model.')

    epochs = Parameter('epochs', 
                    default = 100, 
                    type    = int, 
                    help    ='Number of Epochs')

    generator = Parameter('generator', 
                        default = 'biggan', 
                        type = click.Choice(['biggan', 'dall-e', 'stylegan']),
                        help    = 'Choose what type of generator you would like to use BigGan or Dall-E')

    interpolation = Parameter('interpolation', 
                        default = 10,
                        type    = int, 
                        help    ='Number of elements to be interpolated per second and feed to the model')

    textfile = IncludeFile('textfile', 
                        required= True,
                        help   ='Path for the lyrics text file',
                        default='yannic_lyrics.txt')

    audiofile = IncludeFile(
        'audiofile',
        is_text=False,
        help='Path To Audio File To Use In Video',
        default='Weasle_sample_audio.mp3')
    
    dalle_decoder = IncludeFile(
        'decoder-path',
        is_text=False,
        envvar="DALLE_PATH",\
        default="decoder.pkl",
        help="Path to DALL-E's decoder"
    )

    @batch(cpu=4,memory=8000,image='valayob/musicvideobuilder:0.4')
    @step
    def start(self):
        print(type(self.audiofile))
        from utils import init_textfile        
        text_file = self._to_file(self.textfile.encode())
        self.descs = init_textfile(text_file.name)
        self.next(self.train)
   
    @batch(cpu=4,memory=24000,gpu=1,image='valayob/musicvideobuilder:0.4')
    @step
    def train(self):
        """
        Setup and Train the Model. Store rest in S3. 
        """
        import tempfile
        print("Setting Up Model : ",self.generator)
        model,perceptor = self.setup_models()
        with tempfile.TemporaryDirectory() as model_save_dir:
            templist, model = self.train_video_gen(model,perceptor,model_save_dir)
            self._save_latent_vectors(templist)

        self.model = model.cpu().state_dict() 
        self.perceptor = perceptor.cpu().state_dict()
        self.next(self.inference)
    
    @step
    def inference(self):
        self.lyric_tuples = [(idx1, pt) for idx1, pt in enumerate(self.descs)]
        print(f"Starting : {len(self.lyric_tuples)} Jobs")
        self.next(self.build_video_chunk,foreach='lyric_tuples')
    
    @retry(times=4)
    @batch(cpu=4,memory=7500,image='valayob/musicvideobuilder:0.4')
    @step
    def build_video_chunk(self):
        """
        Run Inference on the latent vectors individually stored. 
        """
        idx,lyric_val = self.input
        self.lyric_idx = idx
        print("Creating Video Chunk file")
        print("Loaded All Latent Vectors")
        model,_ = self.setup_models(with_perceptor=False)
        model.load_state_dict(self.model)
        video_temp_file,write_file_name = self.interpolate_lyric_video(lyric_val,self.lyric_idx, model)
        if video_temp_file is not None:
            self.video_url = self.save_video(write_file_name)
        else:
            self.video_url = video_temp_file 

        self.next(self.video_from_lyrics)
    
    @batch(cpu=4,memory=8000,image='valayob/musicvideobuilder:0.4')
    @step
    def video_from_lyrics(self,inputs):
        from metaflow import S3
        import create_video
        import os 
        import shutil
        import shutil
        import tempfile
        videos = []        
        self.descs = inputs.build_video_chunk.descs
        for input in inputs:
            if input.video_url is not None:
                videos.append((input.lyric_idx,input.video_url))
        
        videos.sort(key=lambda x:x[0])
        # Creating temp dir because we cannot have limited file descriptors. 
        # Moving Files from s3 to tempdir and then concating those in the video 
        with tempfile.TemporaryDirectory() as tmpdirname:
            with S3() as s3:
                write_video = []
                for lyric_idx,video_url in videos:
                    s3_resp = s3.get(video_url)
                    video_file_name = os.path.join(tmpdirname,s3_resp.path.split('/')[-1])
                    shutil.move(s3_resp.path,video_file_name)
                    write_video.append(video_file_name)

            audio_file = self._to_file(self.audiofile)
            video_path = create_video.concatvids(self.descs,\
                                    write_video,\
                                    audio_file.name,\
                                    write_to_path='./')
            
            self.final_video_url = self.save_video(video_path)
        self.next(self.end)
    
    @step
    def end(self):
        print("Done Computation")
   
    def _to_file(self,file_bytes):
        """
        Returns path for a file. 
        """
        import tempfile
        latent_temp = tempfile.NamedTemporaryFile(delete=True)
        latent_temp.write(file_bytes)
        latent_temp.seek(0)
        return latent_temp
        
        

    def interpolate_lyric_video(self,lyric,lyric_idx,model):
        import torch
        from utils import create_image
        import create_video
        from dall_e import  unmap_pixels
        video_temp_list = []
        num_frames = 0
        pt = lyric
        # interpole elements between each image
        print("Interpolating Between Images")
        map_location=torch.device('cpu')
        if torch.cuda.is_available():
            model = model.cuda()
            map_location=torch.device('cuda')
        # get the next index of the descs list, 
        # if it z1_idx is out of range, break the loop
        z1_idx = lyric_idx + 1
        if z1_idx >= len(self.descs):
            return None,None
        current_lyric = pt[1]

        # get the interval betwee 2 lines/elements in seconds `ttime`
        d1 = pt[0]
        d2 = self.descs[z1_idx][0]
        ttime = d2 - d1

        # Load zs from file. 
        zs = torch.load(self._get_latent_vector(lyric_idx),map_location=map_location)
        
        # compute for the number of elements to be insert between the 2 elements
        N = round(ttime * self.interpolation)
        # the codes below determine if the output is list (for biggan)
        # if not insert it into a list 
        print("Generating Images :",N)
        if N == 0:
            return None,None
        if not isinstance(zs, list):
            z0 = [zs]
            z1 = [torch.load(self._get_latent_vector(z1_idx),map_location=map_location)]
        else:
            z0 = zs
            z1 = torch.load(self._get_latent_vector(z1_idx),map_location=map_location)
        
        # loop over the range of elements and generate the images
        image_temp_list = []
        for t in range(N):
            num_frames +=1
            azs = []
            for r in zip(z0, z1):
                z_diff = r[1] - r[0] 
                inter_zs = r[0] + sigmoid(t / (N-1)) * z_diff
                azs.append(inter_zs)

            # Generate image
            with torch.no_grad():
                if self.generator == 'biggan':
                    img = model(azs[0], azs[1], 1).cpu().numpy()
                    img = img[0]
                elif self.generator == 'dall-e':
                    img = unmap_pixels(torch.sigmoid(model(azs[0])[:, :3]).cpu().float()).numpy()
                    img = img[0]
                elif self.generator == 'stylegan':
                    img = model(azs[0])
                image_temp = create_image(img, t, current_lyric, self.generator)
            image_temp_list.append(image_temp)
        if len(image_temp_list) == 0:
            return None,None
        video_temp,write_file_name = create_video.createvid(f'{current_lyric}', image_temp_list, duration=ttime / N)
        return video_temp, write_file_name
        
    
    def _save_latent_vectors(self,latent_vector_files):
        from metaflow import S3
        self.latent_vector_files = []
        with S3(run=self) as s3:
            # Order is important so we serially iterate over the files. 
            for f in latent_vector_files:
                resp = s3.put_files([
                    (f.split('/')[-1],f)
                ])
                self.latent_vector_files.append(
                    resp[0][1]
                )

    def _get_latent_vector(self,vec_idx):
        if vec_idx >= len(self.latent_vector_files):
            return None
        from metaflow import S3
        with S3() as s3:
            return self._to_file(s3.get(self.latent_vector_files[vec_idx]).blob)


    def save_video(self,video_path):
        from metaflow import S3
        with S3(run=self) as s3:
            saved = s3.put_files([
                (video_path.split('/')[-1],video_path)
            ])
            s3_url = saved[0][1]
            return s3_url

    def train_video_gen(self,model,perceptor,save_directory):
        from tqdm import tqdm
        from tqdm.contrib.logging import tqdm_logging_redirect
        import torch
        import clip
        import tempfile
        from utils import train, Pars, create_image, create_outputfolder, init_textfile
        import uuid
        import os 

        def unique_file_name():
            return str(uuid.uuid4())
        # Read the textfile 
        
        # descs - list to append the Description and Timestamps
        # list of temporary PTFiles 
        templist = []
        print("Running the Training Loop")
        if torch.cuda.is_available():
            model = model.cuda()
        # Loop over the description list
        with tqdm_logging_redirect():
            for d in tqdm(self.descs):

                timestamp = d[0]
                line = d[1]
                # stamps_descs_list.append((timestamp, line))
                lats = Pars(gen=self.generator,cuda=torch.cuda.is_available())

                # Init Generator's latents
                if self.generator == 'biggan':
                    par     = lats.parameters()
                    lr      = 0.1#.07
                elif self.generator == 'stylegan':
                    par     = [lats.normu]
                    lr      = .01
                elif self.generator == 'dall-e':
                    par     = [lats.normu]
                    lr      = .1

                # Init optimizer
                optimizer = torch.optim.Adam(par, lr)

                # tokenize the current description with clip and encode the text
                txt = clip.tokenize(line)
                if torch.cuda.is_available():
                    txt = txt.cuda()
                percep = perceptor.encode_text(txt).detach().clone()

                # Training Loop
                for i in range(self.epochs):
                    zs = train(i, model, lats, sideX, sideY, perceptor, percep, optimizer, line, txt, epochs=self.epochs, gen=self.generator)
                # save each line's last latent to a torch file in a temporary directory
                file_path = os.path.join(save_directory,unique_file_name())
                torch.save(zs, file_path) 
                # append it to templist so it can be accessed later
                templist.append(file_path)
        return templist, model 
    
    def setup_models(self,with_perceptor=True):
        import clip 
        if with_perceptor:
            perceptor, preprocess   = clip.load('ViT-B/32')
            perceptor               = perceptor.eval()
        else:
            perceptor = None

        # Load the model
        if self.generator == 'biggan':
            from biggan         import BigGAN
            model   = BigGAN.from_pretrained('biggan-deep-512')
            model   = model.eval()
            
        elif self.generator == 'dall-e':
            from dall_e         import  load_model
            dalle_decoder_file = self._to_file(self.dalle_decoder)
            model   = load_model(dalle_decoder_file.name, 'cpu')
        elif self.generator == 'stylegan':
            from stylegan       import g_synthesis
            model   = g_synthesis.eval()

        return model,perceptor


if __name__ == "__main__":
    VideoGenerationPipeline()