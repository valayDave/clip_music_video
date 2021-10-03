from metaflow import FlowSpec,step,batch,Parameter,IncludeFile
import click
import math

sideX       = 512
sideY       = 512

def sigmoid(x):
    x = x * 2. - 1.
    return math.tanh(1.5*x/(math.sqrt(1.- math.pow(x, 2.)) + 1e-6)) / 2 + .5


class VideoGenerationPipeline(FlowSpec):

    batch_size = Parameter('batch-size',default = 64,type=int,help='Batch size to use for training the model.')

    num_gpus = Parameter(
        'num-gpus',
        envvar="NUM_GPUS",\
        default=0,type=int,help='Number of GPUs to use when training the model.'
    )
    
    epochs = Parameter('epochs', 
                    default = 100, 
                    type    = int, 
                    help    ='Number of Epochs')

    generator = Parameter('generator', 
                        default = 'biggan', 
                        type = click.Choice(['biggan', 'dall-e', 'stylegan']),
                        help    = 'Choose what type of generator you would like to use BigGan or Dall-E')

    use_lyrics = Parameter('lyrics', 
                is_flag=True,
                help    ='Include lyrics')

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
        default="decoder.pkl",
        help="Path to DALL-E's decoder"
    )

    max_frames = Parameter(
        'max-frames', 
        default = None,
        type    = int, 
        help    ='Maximum Number of frames ')

    # @batch(cpu=4,memory=8000,image='valayob/musicvideobuilder:0.3')
    @step
    def start(self):
        print(type(self.audiofile))
        from utils import init_textfile        
        text_file_path = self._to_file(self.textfile.encode())
        self.descs = init_textfile(text_file_path)
        self.next(self.train)
    
    def _to_file(self,file_bytes,as_name=True):
        """
        Returns path for a file. 
        """
        import tempfile
        latent_temp = tempfile.NamedTemporaryFile(delete=False)
        latent_temp.write(file_bytes)
        latent_temp.seek(0)
        if not as_name:
            return latent_temp
        return latent_temp.name
        

    @batch(cpu=4,memory=24000,gpu=1,image='valayob/musicvideobuilder:0.4')
    @step
    def train(self):
        """
        Setup and Train the Model. Store rest in S3. 
        """
        print("Setting Up Model : ",self.generator)
        model,perceptor = self.setup_models()
        templist, model = self.train_video_gen(model,perceptor)
        from metaflow import S3
        with S3(run=self) as s3:
            s3_resp = s3.put_files([
                (f.name.split('/')[-1],f.name) for f in templist
            ])
            self.latent_vector_files = [r[1] for r in s3_resp]

        self.model = model.cpu().state_dict() 
        self.perceptor = perceptor.cpu().state_dict()
        self.next(self.inference)
    
    @batch(cpu=4,memory=24000,gpu=1,image='valayob/musicvideobuilder:0.4')
    @step
    def inference(self):
        """
        Run Inference on the latent vectors stored. 
        """
        from metaflow import S3
        print("Training Complete : Creating Video file")
        with S3() as s3:
            templist =  []
            for ret_obj in s3.get_many(self.latent_vector_files):
                templist.append(self._to_file(ret_obj.blob,as_name=True) )
            model,_ = self.setup_models()
            if self.num_gpus > 0 :
                model = model.cuda()
            model.load_state_dict(self.model)
            audio_file_path = self._to_file(self.audiofile)
            path_to_video = self.interpolate(templist, self.descs, model, audio_file_path)
            self.video_url = self.save_video(path_to_video)
        self.next(self.end)
    
    
    def save_video(self,video_path):
        from metaflow import S3
        with S3(run=self) as s3:
            saved = s3.put_files([
                (video_path.split('/')[-1],video_path)
            ])
            s3_url = saved[0][1]
            return s3_url

    def interpolate(self,templist, descs, model, audiofile):
        import torch
        from utils import create_image
        import create_video
        from dall_e import  unmap_pixels
        video_temp_list = []
        num_frames = 0
        # interpole elements between each image
        print("Interpolating Between Images")
        for idx1, pt in enumerate(descs):
            # get the next index of the descs list, 
            # if it z1_idx is out of range, break the loop
            z1_idx = idx1 + 1
            if z1_idx >= len(descs):
                break

            current_lyric = pt[1]

            # get the interval betwee 2 lines/elements in seconds `ttime`
            d1 = pt[0]
            d2 = descs[z1_idx][0]
            ttime = d2 - d1

            # if it is the very first index, load the first pt temp file
            # if not assign the previous pt file (z1) to zs variable
            if idx1 == 0:
                zs = torch.load(templist[idx1])
            else:
                zs = z1
            
            # compute for the number of elements to be insert between the 2 elements
            N = round(ttime * self.interpolation)
            # the codes below determine if the output is list (for biggan)
            # if not insert it into a list 
            if not isinstance(zs, list):
                z0 = [zs]
                z1 = [torch.load(templist[z1_idx])]
            else:
                z0 = zs
                z1 = torch.load(templist[z1_idx])
            
            # loop over the range of elements and generate the images
            image_temp_list = []
            for t in range(N):
                num_frames +=1
                if self.max_frames is not None:
                    if num_frames > self.max_frames:
                        break
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
            if len(image_temp_list) > 0:
                video_temp = create_video.createvid(f'{current_lyric}', image_temp_list, duration=ttime / N)
                video_temp_list.append(video_temp)
            if self.max_frames is not None:
                if num_frames > self.max_frames:
                    break
            
        # Finally create the final output and save to output folder
        return create_video.concatvids(descs, video_temp_list, audiofile, lyrics=self.use_lyrics,write_to_path='./')
        

    def train_video_gen(self,model,perceptor):
        from tqdm import tqdm
        from tqdm.contrib.logging import tqdm_logging_redirect
        import torch
        import clip
        import tempfile
        from utils import train, Pars, create_image, create_outputfolder, init_textfile
        # Read the textfile 
        
        # descs - list to append the Description and Timestamps
        # list of temporary PTFiles 
        templist = []
        print("Running the Training Loop")
        if self.num_gpus > 0:
            model = model.cuda()
        # Loop over the description list
        with tqdm_logging_redirect():
            for d in tqdm(self.descs):

                timestamp = d[0]
                line = d[1]
                # stamps_descs_list.append((timestamp, line))
                lats = Pars(gen=self.generator,cuda=self.num_gpus > 0)

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
                if self.num_gpus > 0:
                    txt = txt.cuda()
                percep = perceptor.encode_text(txt).detach().clone()

                # Training Loop
                for i in range(self.epochs):
                    zs = train(i, model, lats, sideX, sideY, perceptor, percep, optimizer, line, txt, epochs=self.epochs, gen=self.generator)

                # save each line's last latent to a torch file temporarily
                latent_temp = tempfile.NamedTemporaryFile()
                torch.save(zs, latent_temp) #f'./output/pt_folder/{line}.pt')
                latent_temp.seek(0)
                #append it to templist so it can be accessed later
                templist.append(latent_temp)
        return templist, model 
    
    def setup_models(self):
        import clip 
        from stylegan       import g_synthesis
        from dall_e         import  load_model
        from biggan         import BigGAN
        perceptor, preprocess   = clip.load('ViT-B/32')
        perceptor               = perceptor.eval()

        
        # Load the model
        if self.generator == 'biggan':
            model   = BigGAN.from_pretrained('biggan-deep-512')
            model   = model.eval()
            
        elif self.generator == 'dall-e':
            dalle_decoder_path = self._to_file(self.dalle_decoder)
            model   = load_model(dalle_decoder_path, 'cpu')
        elif self.generator == 'stylegan':
            model   = g_synthesis.eval()

        return model,perceptor

    @step
    def end(self):
        pass


if __name__ == "__main__":
    VideoGenerationPipeline()