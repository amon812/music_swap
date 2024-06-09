import random
from datetime import datetime
import copy
import logging
import time
from pathlib import Path

from moviepy.editor import *
import moviepy.video.fx.all as vfx
import numpy as np
from itertools import chain


from music_swap.util import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



DUMP_INTERMEDIATE_VIDEO = True



def time_stretch_every_beat(clip_orig:VideoFileClip, dbeats, ref_dbeats, clip_length):

    time_stretched_video_slices = []

    def inner(clip_orig, dbeats, ref_dbeats, clip_length):
        nonlocal time_stretched_video_slices

        length=0

        factor = 1
        if len(dbeats) > 1 and len(ref_dbeats) > 1:
            factor = (dbeats[1]-dbeats[0]) / (ref_dbeats[1]-ref_dbeats[0])
        
        logger.debug(f"{factor=}")
        head_len = ref_dbeats[0] * factor
        logger.debug(f"{ref_dbeats[0]=}")
        logger.debug(f"{head_len=}")

        dbeats = dbeats[ dbeats >= head_len ]

        slice = clip_orig.subclip((dbeats[0] - head_len)/44100, dbeats[0]/44100).speedx(factor=factor)
        time_stretched_video_slices.append(slice)

        tail = dbeats[0]/44100

        length += slice.duration

        movie_chunks = []
        for i,(org_b0,org_b1) in enumerate(zip(dbeats,dbeats[1:])):
            movie_chunks.append( (org_b0,org_b1) )
        music_chunks = []
        for i,(ref_b0,ref_b1) in enumerate(zip(ref_dbeats,ref_dbeats[1:])):
            music_chunks.append( (ref_b0,ref_b1) )

        logger.debug(f"{len(movie_chunks)=}")

        for i, ((org_b0,org_b1), (ref_b0,ref_b1)) in enumerate(zip(movie_chunks, music_chunks)):
            factor = (org_b1-org_b0) / (ref_b1-ref_b0)
            logger.debug(f"{factor=}")
            logger.debug(f"{org_b0=}")
            slice = clip_orig.subclip(org_b0/44100, org_b1/44100).speedx(factor=factor)

            time_stretched_video_slices.append(slice)
            tail = org_b1/44100

            length += slice.duration

        # tail
        slice = clip_orig.subclip(tail).speedx(factor=factor)
        time_stretched_video_slices.append(slice)

        length += slice.duration
        
        return length
    
    cur_pos = 0
    cur_ref_dbeats = ref_dbeats

    if len(cur_ref_dbeats) == 0 or len(dbeats) == 0:
        return clip_orig

    length = inner(clip_orig, dbeats, cur_ref_dbeats, clip_length)
    cur_pos += length

    cur_ref_dbeats = [b - cur_pos*44100 for b in ref_dbeats if b >= cur_pos*44100]

    output = concatenate_videoclips(time_stretched_video_slices)
    return output


def time_stretch_every_beat_from_back(clip_orig:VideoFileClip, dbeats, ref_dbeats, scene_length):

    time_stretched_video_slices = []
    logger.info(f"{clip_orig.duration=}")
    logger.info(f"{scene_length=}")

    dbeats = dbeats[::-1]
    ref_dbeats = ref_dbeats[::-1]

    tail_pos = scene_length*44100

    ref_dbeats = [b for b in ref_dbeats if b < tail_pos]

    def inner(clip_orig, dbeats, ref_dbeats, scene_length):
        nonlocal time_stretched_video_slices

        if False:
            tmpstr = "clip_orig" + get_timestr() + "_" + "back" + ".mp4"
            clip_orig.write_videofile(tmpstr)

        length=0

        factor = 1
        if len(dbeats) > 1 and len(ref_dbeats) > 1:
            factor = (dbeats[1]-dbeats[0]) / (ref_dbeats[1]-ref_dbeats[0])

        logger.debug(f"{factor=}")
        last_slice_length =  (tail_pos - ref_dbeats[0]) * factor
        logger.debug(f"{(tail_pos - ref_dbeats[0])=}")
        logger.debug(f"{last_slice_length=}")

        dbeats = dbeats[ dbeats < clip_orig.duration*44100 - last_slice_length ]

        logger.debug(f"st {dbeats[0]/44100}")
        logger.debug(f"end {(dbeats[0] + last_slice_length)/44100}")
        slice = clip_orig.subclip(dbeats[0]/44100, (dbeats[0] + last_slice_length)/44100).speedx(factor=factor)

        # dump video
        if False:
            tmpstr = "slice_dump_" + get_timestr() + "_" + "last" + ".mp4"
            slice.write_videofile(tmpstr)

        time_stretched_video_slices.append(slice)

        head = dbeats[0]/44100
        logger.debug(f"{head=}")

        length += slice.duration

        movie_chunks = []
        for i,(org_b0,org_b1) in enumerate(zip(dbeats,dbeats[1:])):
            movie_chunks.append( (org_b0,org_b1) )
        music_chunks = []
        for i,(ref_b0,ref_b1) in enumerate(zip(ref_dbeats,ref_dbeats[1:])):
            music_chunks.append( (ref_b0,ref_b1) )

        logger.debug(f"{len(movie_chunks)=}")

        for i, ((org_b1,org_b0), (ref_b1,ref_b0)) in enumerate(zip(movie_chunks, music_chunks)):
            factor = (org_b1-org_b0) / (ref_b1-ref_b0)
            logger.debug(f"{factor=}")
            logger.debug(f"{org_b0=}")

            logger.debug(f"st {org_b0/44100}")
            logger.debug(f"end {org_b1/44100}")

            slice = clip_orig.subclip(org_b0/44100, org_b1/44100).speedx(factor=factor)

            time_stretched_video_slices.append(slice)
            head = org_b0/44100
            logger.debug(f"{head=}")

            length += slice.duration

        # head
        logger.debug(f"{head=}")

        logger.debug(f"st {0}")
        logger.debug(f"end {head}")
       
        slice = clip_orig.subclip(0, head).speedx(factor=factor)
        time_stretched_video_slices.append(slice)

        length += slice.duration
        
        return length
    
    cur_ref_dbeats = ref_dbeats

    if len(cur_ref_dbeats) == 0 or len(dbeats) == 0:
        return clip_orig
    
    length = inner(clip_orig, dbeats, cur_ref_dbeats, scene_length)

    time_stretched_video_slices.reverse()

    output = concatenate_videoclips(time_stretched_video_slices)

    if False:
        tmpstr = "concatenated_dump_" + get_timestr() + "_" + "back" + ".mp4"
        output.write_videofile(tmpstr)

    logger.info(f"{length=}")
    logger.info(f"{output.duration=}")

    return output



def modify_beats(master_p_dbeats, slave_p_dbeats, speed):
    if speed == 1.0:
        return master_p_dbeats, slave_p_dbeats
    
    def double_beats(beats):
        if beats is None:
            return None

        tmp = []
        for b0,b1 in zip(beats, beats[1:]):
            tmp.append( b0 + (b1-b0)/2 )
        tmp = np.sort( np.concatenate( [beats, np.array(tmp, dtype=int)] ) )
        return tmp
    
    def half_beats(beats):
        if beats is None:
            return None
        
        tmp = [beats[i] for i in range(0, len(beats), 2)]
        tmp = np.array(tmp, dtype=int)
        return tmp
    
    if speed < 0.5:
        return double_beats(double_beats(master_p_dbeats)), slave_p_dbeats
    elif speed < 1.0:
        return double_beats(master_p_dbeats), slave_p_dbeats
    elif speed < 2.0:
        return master_p_dbeats, slave_p_dbeats
    elif speed < 4.0:
        return half_beats(master_p_dbeats), slave_p_dbeats
    else:
        return half_beats(half_beats(master_p_dbeats)), slave_p_dbeats

def randomize_shots(clip_info, master_p_dbeats):

    clip = clip_info["clip"]

    master_p_dbeats = [m/44100 for m in master_p_dbeats]

    #min_length = (master_p_dbeats[1]-master_p_dbeats[0])*4 * 2
    min_length = clip_info["scene_detection_min_sec"]
    logger.debug(f"detect_scene {min_length=}")

    scene_detection_threshold = clip_info["scene_detection_threshold"]

    scene_ranges = detect_scene(clip_info["file_path"], clip_info["start_sec"], clip_info["end_sec"], min_length * clip.fps, scene_detection_threshold )
    logger.debug(f"{scene_ranges=}")

    if clip_info["random_shots"]:
        random.shuffle(scene_ranges)

    if clip_info["repeat_shots"]:
        r = clip_info["repeat_shots_range"]

        scene_ranges = [ [a] * random.randint(r[0],r[1]) for a in scene_ranges]
        scene_ranges = list(chain.from_iterable(scene_ranges))


    logger.debug(f"{scene_ranges=}")

    clip_list = []
    new_beats = []

    cur_pos = 0

    for s in scene_ranges:
        beats = [b for b in master_p_dbeats if s[0] <= b < s[1]]
        if len(beats) > 1:
            sample_length = beats[1] - beats[0]
            if sample_length / 2 <= (beats[0] - s[0]):
                mod_s0 = beats[0] - (sample_length / 2)
            elif sample_length / 3 > (beats[0] - s[0]):
                mod_s0 = beats[0] + (sample_length / 2)
            else:
                mod_s0 = s[0]
            
            if sample_length / 2 <= (s[1] - beats[-1]):
                mod_s1 = beats[-1] + (sample_length / 2)
            elif sample_length / 3 > (s[1] - beats[-1]):
                mod_s1 = beats[-1] - (sample_length / 2)
            else:
                mod_s1 = s[1]
        else:
            mod_s0, mod_s1 = s[0], s[1]

        tmp_clip = clip.subclip(mod_s0, mod_s1)
        clip_list.append( tmp_clip )
        beats = [b-mod_s0 for b in master_p_dbeats if mod_s0 <= b < mod_s1]
        beats = [b+cur_pos for b in beats]
        logger.debug(f"mod {beats=}")


        new_beats += beats
        cur_pos += tmp_clip.duration

    logger.debug(f"{new_beats=}")

    clip = concatenate_videoclips(clip_list)

    return clip, np.array([ int(b*44100) for b in new_beats] , dtype=int)






def swap_stretch_video_scene(clip_info_scene, slave_p_dbeats, total_cur_pos, scene_length, is_last_scene):

    logger.debug(f"{is_last_scene=}")

    remaining_length = scene_length

    audio_list=[]
    last_clip = None

    def fadeinout(clip, c):
        fadein = c.get( "fadein", 0 )
        fadeout = c.get( "fadeout", 0 )
        fade_color = c.get( "fade_color", [0,0,0] )

        if fadein not in [0,-1]:
            clip = clip.fadein(fadein, fade_color)
            clip = clip.audio_fadein(fadein)
        if fadeout not in [0,-1]:
            clip = clip.fadeout(fadeout, fade_color)
            clip = clip.audio_fadeout(fadeout)
        return clip
    def audio_fadeinout(clip, c):
        fadein = c.get( "fadein", 0 )
        fadeout = c.get( "fadeout", 0 )

        if fadein not in [0,-1]:
            clip = clip.audio_fadein(fadein)
        if fadeout not in [0,-1]:
            clip = clip.audio_fadeout(fadeout)
        return clip
    
    if clip_info_scene[-1]["fill_from_back"]:
        c = clip_info_scene.pop()

        if c["use_once"] == False:
            clip_info_scene.append(c)

        clip = c["clip"]
        audio = clip.audio

        speed = c.get("speed", 1.0)

        slave_p_dbeats = [[b for b in bs if b < remaining_length*44100] for bs in slave_p_dbeats]

        if c["stretch"]:
            master_p_dbeats = c["master_beats"]

            cur_slave_p_dbeats = slave_p_dbeats[ c["stretch_method"] ]

            master_p_dbeats, cur_slave_p_dbeats = modify_beats(master_p_dbeats, cur_slave_p_dbeats, speed)

            if c["random_shots"] or c["repeat_shots"]:
                clip, master_p_dbeats = randomize_shots(c, master_p_dbeats)

            clip = time_stretch_every_beat_from_back(clip, master_p_dbeats, cur_slave_p_dbeats, remaining_length)
        else:
            if speed != 1.0:
                clip = clip.speedx(factor=speed)

        if remaining_length < clip.duration:
            clip = clip.subclip(clip.duration - remaining_length)
        
        remaining_length -= clip.duration

        if c["use_audio"]:
            audio = audio.subclip(0,clip.duration)
            audio = audio.set_start( total_cur_pos + remaining_length)
            vol = c.get("volume", 1.0)
            if vol != 1.0:
                audio = audio.volumex(vol)
            audio = audio_fadeinout(audio, c)
            audio_list.append(audio)

        clip = fadeinout(clip, c)

        last_clip = clip

    slave_p_dbeats = [[b for b in bs if b < remaining_length*44100] for bs in slave_p_dbeats]

    clip_list=[]
    cur_pos = 0
    
    for i, c in enumerate(clip_info_scene):

        if c["use_once"] == False:
            clip_info_scene.append(c)

        logger.info(f"{i=} : {remaining_length=}")
        if remaining_length <= 0:
            break

        clip = c["clip"]
        audio = clip.audio

        speed = c.get("speed", 1.0)

        if c["stretch"]:
            master_p_dbeats = c["master_beats"]

            cur_slave_p_dbeats = slave_p_dbeats[ c["stretch_method"] ]
            
            cur_slave_p_dbeats = [b - cur_pos*44100 for b in cur_slave_p_dbeats if b >= cur_pos*44100]

            master_p_dbeats, cur_slave_p_dbeats = modify_beats(master_p_dbeats, cur_slave_p_dbeats, speed)

            if c["random_shots"] or c["repeat_shots"]:
                clip, master_p_dbeats = randomize_shots(c, master_p_dbeats)

            if len(cur_slave_p_dbeats) > 0:
                clip = time_stretch_every_beat(clip, master_p_dbeats, cur_slave_p_dbeats, remaining_length)
            else:
                if remaining_length < clip.duration:
                    clip = clip.subclip(0, remaining_length)
        else:
            if speed != 1.0:
                clip = clip.speedx(factor=speed)


        logger.debug(f"{i=} : {remaining_length=}")
        logger.debug(f"{i=} : {clip.duration=}")
        
        if remaining_length < clip.duration:
            clip = clip.subclip(0, remaining_length)
        
        clip = fadeinout(clip, c)

        clip_list.append(clip)

        if c["use_audio"]:
            audio = audio.subclip(0,clip.duration)
            audio = audio.set_start(total_cur_pos + cur_pos)
            vol = c.get("volume", 1.0)
            if vol != 1.0:
                audio = audio.volumex(vol)
            audio = audio_fadeinout(audio, c)
            audio_list.append(audio)
        
        cur_pos += clip.duration

        remaining_length -= clip.duration
    
    if is_last_scene and not last_clip:
        pass
    elif remaining_length > 0:
        logger.info(f"blank {remaining_length=}")
        blank = ColorClip( (clip.w, clip.h), color=(0,0,0), duration=  remaining_length )
        clip_list.append(blank)
    
    if last_clip:
        clip_list.append(last_clip)
        

    final_clip = concatenate_videoclips(clip_list)

    return final_clip, audio_list

def create_method2_beats(audio_file_path, clip_info_list, org_beats_list):
    method2_exist = False
    def stretch_method_check(v):
        nonlocal method2_exist
        if v == 2:
            method2_exist = True
        return v

    update_key({"t":clip_info_list}, "stretch_method", stretch_method_check)

    if not method2_exist:
        return None
    
    song = Song(audio_file_path)
    downbeats = song.get_downbeats()
    downbeats = np.array([d/44100 for d in downbeats])    
    
    boundary = analyze_music_boundary(audio_file_path, downbeats)

    mt2_beats = []

    for b0,b1 in zip(boundary,boundary[1:]):
        if b0[1] == 0:
            # onset
            arr = org_beats_list[1]
        else:
            arr = org_beats_list[0]
        
        start = b0[0]*44100
        end = b1[0]*44100

        if len(mt2_beats) > 0:
            last = mt2_beats[-1]
            start = max(start, last + 0.25 * 44100)

        mt2_beats = np.append(mt2_beats, arr[np.where((start <= arr) & (arr < end))])
    
    logger.debug(mt2_beats)

    return np.array(mt2_beats, dtype=int)






def swap_stretch_video(clip_info_list, scenes, new_audio_file_path):
    random.seed()

    slave_song = Song2(new_audio_file_path)

    slave_p_dbeats = [ slave_song.get_beats44(), slave_song.get_onsets() ]

    mt2_beats = create_method2_beats(new_audio_file_path, clip_info_list, slave_p_dbeats)
    if mt2_beats is not None:
        slave_p_dbeats.append( mt2_beats )

    clip_list = []
    main_audio = AudioFileClip(new_audio_file_path)
    audio_list = [ main_audio ]
    cur_pos = 0

    scene_length_list = []

    scenes = [s for s in scenes if s < main_audio.duration ]

    scenes.append(main_audio.duration)

    for s0,s1 in zip(scenes, scenes[1:]):
        scene_length_list.append(s1-s0)
    
    scene_nums = min( len(clip_info_list), len(scene_length_list) )

    for i, (clip_info_scene, scene_length) in enumerate(zip(clip_info_list, scene_length_list)):

        cur_slave_p_dbeats = [[b - cur_pos*44100 for b in bs if b >= cur_pos*44100] for bs in slave_p_dbeats]

        is_last_scene = (i == (scene_nums-1))

        scene_clip, scene_audio_list = swap_stretch_video_scene(clip_info_scene, cur_slave_p_dbeats, cur_pos, scene_length, is_last_scene)

        # debug dump video
        if DUMP_INTERMEDIATE_VIDEO:
            tmpstr = "swap_stretch_video_" + get_timestr() + "_" + str(i) + ".mp4"
            debug_dump_video(scene_clip, tmpstr)


        cur_pos += scene_clip.duration

        clip_list.append(scene_clip)

        audio_list += scene_audio_list


    if len(clip_list) == 1:
        clip = clip_list[0]
    else:
        clip = concatenate_videoclips(clip_list)


    audio = CompositeAudioClip(audio_list)

    logger.info(f"{clip.duration=}")
    logger.info(f"{audio.duration=}")

    if clip.duration < (audio.duration + audio.start):
        audio = audio.subclip(0, clip.duration - audio.start)
    elif clip.duration > (audio.duration + audio.start):
        logger.error(f"clip.duration > (audio.duration + audio.start)")
        clip = clip.subclip(0, audio.duration + audio.start)
    else:
        pass

    clip.audio = audio

    return clip



###############################################################

def exec_command(clip_info, scenes, new_audio, base_speed=1.0):
    logger.info(f"{clip_info=}")
    logger.info(f"{scenes=}")
    logger.info(f"{new_audio=}")
    logger.info(f"{base_speed=}")

    if base_speed != 1.0:
        new_audio = create_tmp_speed_audio(new_audio, base_speed)

    final_clip = swap_stretch_video(clip_info_list=clip_info, scenes=scenes, new_audio_file_path=new_audio)

    if base_speed != 1.0:
        os.remove(new_audio)
    
    return final_clip

