from datetime import datetime
import json
import logging
import time
from pathlib import Path
import io

import fire
import numpy as np

from moviepy.editor import *
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent, detect_silence
from pydub import effects
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_timestr():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_non_silence_range(file, min_silence_len = 1000, silence_thresh = -40, th_too_short = 250, th_too_long = 5000):
    
    sound = AudioSegment.from_file(file)

    sound = effects.normalize(sound)
    
    logger.debug(f"{sound.dBFS=}")

    raw_ranges = detect_nonsilent(sound, min_silence_len, silence_thresh)
    
    time_ranges = []
    
    # Split larger chunks roughly into pieces.
    # Remove any chunks that are too small( <= 150 ).
    for range in raw_ranges:
        if (range[1] - range[0]) > th_too_long:
            logger.debug(f"{range=} / {range[1] - range[0]} msec" )
            mini_ranges = detect_nonsilent(sound[range[0]:range[1]], min_silence_len // 3 , silence_thresh)
            
            pending = -1
            tail = -1
            
            for mini in mini_ranges:
                logger.debug( f"{mini=}" )
                if ((mini[1] - mini[0]) < th_too_short):
                    # Attach smaller pieces together.
                    logger.debug( f"pend small {mini}" )
                    pending = mini[0] if pending == -1 else pending
                else:
                    start_tim = mini[0] if pending == -1 else pending
                    time_ranges.append( [ start_tim + range[0], mini[1] + range[0]] )
                    pending = -1
                tail = mini[1]
            
            if pending != -1:
                # If the tail is small
                time_ranges.append( [ pending + range[0], tail + range[0]] )
        
        elif (range[1] - range[0]) > 150:
            time_ranges.append(range)
    
    
    # Split the large chunks that still remain somewhat forcefully.
    time_ranges2 = []
    
    for range in time_ranges:
        if (range[1] - range[0]) > 15000:
            logger.debug( f"large chunk {range} -> {range[1] - range[0]} msec" )
            
            mini_ranges = split_chunk(sound[range[0]: range[1]], min_silence_len // 3 , silence_thresh // 1.5)
            
            for mini in mini_ranges:
                logger.debug( f"{mini=} -> {mini[1] - mini[0]} msec" )
                time_ranges2.append( [ mini[0] + range[0], mini[1] + range[0]] )
            
        else:
            time_ranges2.append(range)
    
    
    return time_ranges2


def split_chunk(chunk, min_silence_len = 1000, silence_thresh = -40):
    
    time_ranges = []
    
    if len(chunk) < 10000:
        time_ranges.append([0,len(chunk)])
        return time_ranges
    
    margin = 2500
    
    # calc middle
    silents = detect_silence(chunk[margin:-margin], min_silence_len , silence_thresh)
    
    if len(silents)==0:
        time_ranges.append([0,len(chunk)])
        return time_ranges
    
    logger.debug( f"before {silents}")
    silents = sorted(silents, key=lambda x: x[1]-x[0] , reverse=True)
    logger.debug( f"after {silents}")
    middle_silent = silents[0]
    
    middle = middle_silent[0] + ((middle_silent[1] - middle_silent[0])//2)
    middle += margin
    
    logger.debug( f"{middle=}")
    
    first_half = split_chunk(chunk[:middle-1], min_silence_len, silence_thresh)
    
    time_ranges += first_half
    
    second_half = split_chunk(chunk[middle:], min_silence_len, silence_thresh)
    
    logger.debug( f"{second_half=}")
    
    for range in second_half:
        logger.debug( f"{range=}")
        time_ranges.append([ range[0] + middle, range[1] + middle ])
    
    return time_ranges


def create_thumb(t, wav_file, video:VideoFileClip):
    thumb = video.get_frame(t/1000)
    thumb = Image.fromarray(thumb)
    thumb_path = Path(wav_file).with_suffix('.png')
    thumb.save(thumb_path)



###############################################################

def exec_command(org_audio, ref_video):
    logger.info(f"{org_audio=}")
    logger.info(f"{ref_video=}")

    range_list = detect_non_silence_range(org_audio, min_silence_len=1000, silence_thresh=-40, th_too_short=250, th_too_long=5000)

    sound = AudioSegment.from_file(org_audio)

    output_dir = Path("wav_" + get_timestr() + "_" + Path(org_audio).stem)
    output_dir.mkdir()

    wav_list=[]

    for i, range in enumerate(range_list):
        file_name = "output" + '{:06d}'.format(i) +".wav"
        chunk = sound[range[0]:range[1]]
        out_path = str(output_dir) + "/" + file_name
        chunk.export(out_path, format="wav")
        wav_list.append( (range[0], out_path) )
    
    if ref_video:
        video = VideoFileClip(ref_video)
        for t, wav in wav_list:
            create_thumb( t, wav, video )


class Command:
    def parse(self, org_audio, ref_video=None):

        start_tim = time.time()

        exec_command(org_audio, ref_video)

        logger.info(f"Elapsed time : {time.time() - start_tim}")



fire.Fire(Command)
