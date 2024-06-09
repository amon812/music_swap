import json
import logging
import time
from pathlib import Path

import fire
from moviepy.editor import *
import moviepy.video.fx.all as vfx


from music_swap.add_se import exec_se_command
from music_swap.swap import exec_command
from music_swap.generate_draft import exec_generate_draft_command
from music_swap.util import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

OGG_TO_WAV = True

CLERA_CACHE = False


###############################################################
def create_tmp_audio(info):

    file_path = info["file_path"]

    start_sec = info.get( "start_sec", 0 )
    end_sec = info.get( "end_sec", -1 )
    fadein = info.get( "fadein", 0 )
    fadeout = info.get( "fadeout", 0 )
    vol = info.get( "volume", 1.0 )

    if OGG_TO_WAV:
        file_path = ogg_to_wav(file_path)

    if start_sec == 0:
        if end_sec == -1:
            if vol == 1.0:
                return False, file_path

    audio = AudioFileClip(file_path)

    if end_sec == -1:
        audio = audio.subclip(start_sec)
    else:
        audio = audio.subclip(start_sec, end_sec)
    
    if vol != 1.0:
        audio = audio.volumex(vol)

    tmpstr = "tmp_audio_" + get_timestr() + ".wav"
    audio.write_audiofile(tmpstr)

    return True, tmpstr

def create_tmp_video(video_info_array, resolution=()):

    def create_video(info):
        nonlocal resolution
        file_path = info["file_path"]
        start_sec = info.get( "start_sec", 0 )
        end_sec = info.get( "end_sec", -1 )

        if resolution[0] == -1 or resolution[1] == -1:
            video = VideoFileClip(file_path)
            resolution = (video.h, video.w)
        else:
            video = VideoFileClip(file_path, target_resolution=resolution)

        if end_sec == -1:
            video = video.subclip(start_sec)
        else:
            video = video.subclip(start_sec, end_sec)
        
        return video
    
    song_map={}
    def create_beats(info):
        nonlocal song_map

        file_path = info["file_path"]
        
        if song_map.get(file_path, None) == None:
            song_map[file_path] = Song2(file_path, skip_onsets=True)
        song = song_map[file_path]
        beats = song.get_beats44()

        start_sec = info.get( "start_sec", 0 )
        end_sec = info.get( "end_sec", -1 )

        if end_sec != -1:
            return np.array([ b - start_sec*44100 for b in beats if start_sec*44100 <= b < end_sec*44100 ])
        else:
            return np.array([ b - start_sec*44100 for b in beats if start_sec*44100 <= b ])


    
    all_clip_info_list=[]

    for i, info in enumerate(video_info_array):
        if type(info) != list:
            info = [info]
        
        clip_info_list = []
        
        for unit in info:
            video = create_video(unit)

            option = unit.get( "option", {} ),
            if type(option) == tuple:
                option = option[0]

            logger.debug(f"{option=}")

            clip_info ={
                "clip" : video,

                "master_beats" : create_beats(unit),

                "file_path" : unit.get("file_path"),
                "start_sec" : unit.get( "start_sec", 0 ),
                "end_sec" : unit.get( "end_sec", -1 ),
                "length" : unit.get( "length", -1 ),

                "stretch" : unit.get( "stretch", True ),

                "stretch_method" : option.get( "stretch_method", 2 ),
                "use_once" : option.get( "use_once", False ),

                "fadein" : option.get( "fadein", 0 ),
                "fadeout" : option.get( "fadeout", 0 ),
                "fade_color" : option.get( "fade_color", [0,0,0] ),
                "use_audio" : option.get( "use_audio", False ),
                "volume" : option.get( "volume", 1.0 ),
                "speed" : option.get( "speed", 1.0 ),

                "random_shots" : option.get( "random_shots", False ),
                "repeat_shots" : option.get( "repeat_shots", False ),
                "repeat_shots_range" : option.get( "repeat_shots_range", [1,2] ),
                "scene_detection_threshold" : option.get( "scene_detection_threshold", 50 ),
                "scene_detection_min_sec" : option.get( "scene_detection_min_sec", 4.0 ),

                "fill_from_back" : option.get( "fill_from_back", False),
            }

            if clip_info["stretch_method"] not in [0,1,2]:
                raise ValueError(f"'stretch_method' must be 0, 1 or 2. {clip_info['stretch_method']=}")

            clip_info_list.append(clip_info)
        
        all_clip_info_list.append(clip_info_list)

    return all_clip_info_list, resolution
    




###############################################################

class Command:
    def _test(self, movie):
        scenes = detect_scene(movie, 0, -1, 0.2, 50)

        scenes = [s[0] for s in scenes]

        scenes = np.array(scenes)

        boundary = analyze_music_boundary(movie, scenes)
        boundary.append((-1, -1))

        video = VideoFileClip(movie)

        test_dir = Path("test_boundary_scene_" + get_timestr())
        test_dir.mkdir(exist_ok=True)

        for i,(b0,b1) in enumerate(zip(boundary,boundary[1:])):
            tmpstr = str(test_dir / Path(str(i).zfill(3) + "_" + "F" + str(b0[1]) + ".mp4"))
            if b1[0] != -1:
                debug_dump_video(video.subclip( b0[0], b1[0] ), tmpstr)
            else:
                debug_dump_video(video.subclip( b0[0] ), tmpstr)


    def generate_draft(self, video, new_audio, video_start=0, video_end=-1):

        if CLERA_CACHE:
            clear_cache()
        create_all_dirs()

        start_tim = time.time()

        GlobalOption.load("generate_draft_template.json")

        info_all = {}

        with open("generate_draft_template.json", "r") as f:
            info_all = json.load(f)

        logger.info(f"{video=}")
        logger.info(f"{new_audio=}")
        logger.info(f"{video_start=}")
        logger.info(f"{video_end=}")


        if OGG_TO_WAV:
            new_audio = ogg_to_wav(new_audio)

        result_path = get_outputfile_name(video, new_audio, ".json")
        
        info_all = exec_generate_draft_command(info_all, video, new_audio, video_start, video_end)

        json_text = json.dumps(info_all, indent=4, ensure_ascii=False)
        json_text = json_text.replace("\r\n","\n").replace("\n","\r\n")
        Path(result_path).write_text(json_text, encoding="utf-8")

        logger.info(f"Output : {result_path}")
        logger.info(f"Elapsed time : {time.time() - start_tim}")
        logger.info(f"python main.py swap2 \"{result_path}\"")



    def swap1(self, video, new_audio, base_speed=1.0, random=False, repeat=False):

        GlobalOption.load("swap1_template.json")

        info_all = {}

        with open("swap1_template.json", "r") as f:
            info_all = json.load(f)
        
        info_all["video"][0][0]["file_path"] = video
        info_all["video"][0][0]["option"]["random_shots"] = random
        info_all["video"][0][0]["option"]["repeat_shots"] = repeat
        info_all["audio"]["file_path"] = new_audio
        info_all["base_speed"] = base_speed

        self._swap(info_all)


    def swap2(self, json_file):

        logger.info(f"{json_file=}")

        GlobalOption.load(json_file)

        info_all = {}

        with open(json_file, "r") as f:
            info_all = json.load(f)
        
        self._swap(info_all)

    def _swap(self, info_all):

        if CLERA_CACHE:
            clear_cache()
        create_all_dirs()

        start_tim = time.time()

        update_key(info_all, "start_sec", to_seconds)
        update_key(info_all, "end_sec", to_seconds)
        update_key(info_all, "boundary", to_seconds)

        logger.debug(info_all)

        if not info_all:
            return
        
        resolution = (info_all["resolution"][1], info_all["resolution"][0])
        
        def swap( info ):
            nonlocal resolution

            clip_info, resolution = create_tmp_video(info["video"], resolution)
            
            tmp_created, audio_path = create_tmp_audio(info["audio"])

            logger.info(f"before exec_command : {time.time() - start_tim}")

            scenes= info["audio"].get("boundary", [])

            end_sec = info["audio"].get("end_sec", -1)
            if end_sec != -1:
                scenes = [s for s in scenes if s < end_sec]

            start_sec = info["audio"].get("start_sec", 0)
            scenes = [s-start_sec for s in scenes]
            scenes = [s for s in scenes if s >= 0]

            scenes = [s / info["base_speed"] for s in scenes]

            if len(scenes) == 0:
                scenes=[0]
            else:
                if scenes[0] != 0:
                    scenes = [0] + scenes
    
            result_clip = exec_command(clip_info=clip_info, scenes=scenes, new_audio=audio_path, base_speed=info["base_speed"])

            logger.info(f"after exec_command : {time.time() - start_tim}")

            if tmp_created:
                os.remove(audio_path)
            
            #fadein
            fadein = info["audio"]["fadein"]
            if fadein not in (0, -1):
                result_clip = result_clip.audio_fadein(fadein)

            #fadeout
            fadeout = info["audio"]["fadeout"]
            if fadeout not in (0, -1):
                result_clip = result_clip.audio_fadeout(fadeout)
            
            return result_clip

        head_info_video = info_all["video"][0]
        if type(head_info_video) == list:
            head_info_video = head_info_video[0]

        output_path = get_outputfile_name(head_info_video["file_path"] , info_all["audio"].get("file_path", "no_audio") )

        clip_list = []

        clip_list.append( swap(info_all) )

        encode_option = info_all.get("encode_option", DefaultEncodingOption)

        output_path = combine_clips(clip_list , output_path, encode_option)

        backup_json_path = Path(output_path).with_suffix(".json")
        json_text = json.dumps(info_all, indent=4, ensure_ascii=False)
        json_text = json_text.replace("\r\n","\n").replace("\n","\r\n")
        backup_json_path.write_text(json_text, encoding="utf-8")

        logger.info(f"Output : {output_path}")
        
        logger.info(f"Elapsed time : {time.time() - start_tim}")


    def add_se(self, json_file):

        if CLERA_CACHE:
            clear_cache()
        create_all_dirs()

        start_tim = time.time()

        GlobalOption.load(json_file)

        logger.info(f"{json_file=}")

        info_se = {}

        with open(json_file, "r") as f:
            info_se = json.load(f)
        
        update_key(info_se, "start_sec", to_seconds)
        update_key(info_se, "end_sec", to_seconds)

        logger.info(info_se)

        if not info_se:
            return
        
        result_clip = exec_se_command(info_se["video_file_path"], info_se["sound_effect"])

        new_path = get_outputfile_name(info_se["video_file_path"], "add_se")

        encode_option = info_se.get("encode_option", DefaultEncodingOption)

        new_path = encode_video(result_clip, new_path, encode_option)

        logger.info(f"Output : {new_path}")
        logger.info(f"Elapsed time : {time.time() - start_tim}")        



fire.Fire(Command)
