from textwrap3          import fill
import moviepy.editor   as me
import tempfile
import textwrap
import glob
import os



def createvid(description, image_temp_list, fps=24, duration=0.1):
    blackbg = me.ColorClip((720,720), (0, 0, 0))

    clips = [me.ImageClip(m.name+".png", duration=duration) for m in image_temp_list]
    for img in image_temp_list:
        img.close()
    concat_clip = me.concatenate_videoclips(clips, method="compose").set_position(('center', 'center'))
    if description == "start song":
        description = " "
    if len(description) > 35:
        description = fill(description, 35)

    txtClip = me.TextClip(description, color='white', fontsize=30, font='Amiri-regular').set_position('center')
    txt_col = txtClip.on_color(size=(blackbg.w, txtClip.h + 10),
                               color=(0,0,0), pos=('center', 'center'), 
                               col_opacity=0.8)

    txt_mov = txt_col.set_position((0, blackbg.h-20-txtClip.h))
    comp_list = [blackbg, concat_clip, txt_mov]
    final = me.CompositeVideoClip(comp_list).set_duration(concat_clip.duration)

    video_tempfile = tempfile.NamedTemporaryFile(delete=False)
    write_file_name = video_tempfile.name+".mp4"
    final.write_videofile(write_file_name, fps=fps)
    video_tempfile.seek(0)

    for clip in clips:
        clip.close()
    for clip in comp_list:
        clip.close()
    return video_tempfile,write_file_name

def concatvids(descriptions, \
            video_temp_list_paths, \
            audiofilepath, \
            fps=24, \
            lyrics=True,\
            write_to_path=None):
    clips = []

    for idx, (desc, vid) in enumerate(zip(descriptions, video_temp_list_paths)):
        if desc == descriptions[-1][1]:
            break
        vid = me.VideoFileClip(f'{vid}')#.set_position(('center', 'center'))
        clips.append(vid)

    concat_clip = me.concatenate_videoclips(clips, method="compose").set_position(('center', 'center'))
    # concat_clip = me.CompositeVideoClip([blackbg, concat_clip])#.set_duration(vid.duration)
    if audiofilepath:
        concat_clip.audio = me.AudioFileClip(audiofilepath)

        concat_clip.duration = concat_clip.audio.duration
    write_path = None
    if write_to_path is None:
        write_path = os.path.join('output', f"finaloutput.mp4")
    else:
        write_path = os.path.join(write_to_path, f"finaloutput.mp4") 
    concat_clip.write_videofile(write_path, fps=fps)
    return write_path




