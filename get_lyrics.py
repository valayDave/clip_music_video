import click
import datetime

def transcript_to_flow_format(transcript):
    save_lines = []
    for line in transcript:
        timing = int(line['start']+line['duration'])
        timestamp = ':'.join(str(datetime.timedelta(seconds=timing)).split(':')[1:])
        text= line['text'].replace('\n',' ').replace('♪','')
        save_lines.append(f"{timestamp} {text}")
    return '\n'.join(save_lines)

@click.command(help='Create Lyrics compatible with Flow using Youtube API')
@click.argument('video_id')
@click.option('-l','--lang',default=['en'], multiple=True)
@click.option('-s','--save-file',default=None)
def make_lyrics(video_id,lang=None,save_file=None):
    from youtube_transcript_api import YouTubeTranscriptApi,NoTranscriptFound
    transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=lang)
    if save_file is None:
        save_file = f'{video_id}.txt'
    with open(save_file,'w') as f: 
        f.write(transcript_to_flow_format(transcript))

if __name__ == "__main__":
    make_lyrics()