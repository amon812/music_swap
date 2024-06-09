from datetime import datetime
import logging
import time
import random
import json
from pathlib import Path
from itertools import chain,combinations

import numpy as np

from moviepy.editor import *
import moviepy.video.fx.all as vfx
import madmom

from music_swap.song import Song

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


DefaultEncodingOption={
    "codec":"libx264",      # ["libx264","mpeg4","rawvideo"]
    "preset":"ultrafast",      # ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"]
    "crf": 15,
}

class Option:
    def load(self, json_path):
        with open(json_path, "r") as f:
            info_all = json.load(f)
        
        global_option = info_all.get("global_option", {})
        self.extra_beat_mode = global_option.get("extra_beat_mode", 1)
        self.onset_limit_per_unit = global_option.get("onset_limit_per_unit", 3)


GlobalOption = Option()



class __StereoSong(Song):
    def __init__(self, filepath=None, annotate=True):
        self.annotate = annotate
        self.filepath = filepath
        self.audio = None
        self.sample_rate = 44100
        self.beats = None
        self.downbeats = None

        if filepath is not None:
            self.load_song_audio()
            if annotate == True:
                self.load_beats()


    def load_song_audio(self):
        from essentia.standard import AudioLoader
        self.audio,self.sample_rate,numberChannels,md5,bit_rate,codec = AudioLoader(filename=self.filepath)()
        self.sample_rate = int(self.sample_rate)

    def save_song_audio(self, new_filepath):
        from essentia.standard import AudioWriter

        new_format = new_filepath.split('/')[-1].split('.')[1]

        AudioWriter(filename=new_filepath, bitrate=320, sampleRate=self.sample_rate, format=new_format)(self.audio)

    def speed_change(self, factor):
        self.audio = time_stretch(self.audio, factor, sample_rate=self.sample_rate)


class Song2(Song):
    def __init__(self, filepath=None, skip_onsets=False):
        self.filepath = filepath
        self.audio = None
        self.sample_rate = 44100
        self.beats = None
        self.beats44 = None
        self.downbeats = None
        self.raw_onsets = None
        self.onsets = None

        if filepath is not None:
            self.song_name, self.song_format = self.get_song_name_and_format()
            self.load_song_audio()
            self.load_beats()
            if not skip_onsets:
                self.load_onsets()

    def annotate_onsets(self, output_filepath):
        if True:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(self.filepath)
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))
            tmp_file = "percussive_" + get_timestr() + ".wav"

            sf.write(tmp_file, y_percussive, sr, 'PCM_24')
        if False:
            from pydub import AudioSegment
            song = AudioSegment.from_file(self.filepath)
            bass = song.low_pass_filter(50)
            tmp_file = "percussive_" + get_timestr() + ".wav"
            bass.export(tmp_file)

        downbeats_proc = madmom.features.OnsetPeakPickingProcessor(combine=0.1, fps=100)
        #downbeats_proc = madmom.features.OnsetPeakPickingProcessor(threshold=0.5, combine=0.15, fps=100)
        activations = madmom.features.RNNOnsetProcessor()(tmp_file)
        #activations = madmom.features.RNNOnsetProcessor()(self.filepath)
        beats = downbeats_proc(activations)
        np.savetxt(output_filepath, beats, newline="\n")

        os.remove(tmp_file)

        return beats

    def get_onsets(self):
        if self.onsets is not None:
            return self.onsets
        
        logger.info("onset start")
        start_tim = time.time()
        
        extra_beat_mode = GlobalOption.extra_beat_mode
        onset_limit = GlobalOption.onset_limit_per_unit

        def onset_sample(onsets, num, head, tail):
            bb = set(combinations(onsets, num))

            def sort_func(x):
                sum = 0
                x = (head,) + x + (tail,)
                for i in range(num+1):
                    sum += (x[i+1] - x[i])
                return sum

            return sorted(bb, key=sort_func ,reverse = True)[0]


        if extra_beat_mode == 0:
            beats = self.beats
            dbeats = []

            for beat_sec, beat_num in beats:
                if int(beat_num) in (1,):
                    dbeats.append(beat_sec)

            onsets =[]
            for b0, b1 in zip(dbeats,dbeats[1:]):
                start = b0
                end = b1
                onset_unit = [o for o in self.raw_onsets if start<= o < end]
                if len(onset_unit) > onset_limit:
                    onset_unit = onset_sample(onset_unit, onset_limit, start, end)
                onsets.append( onset_unit )

            onsets = list(chain.from_iterable(onsets))
            dbeats += onsets
            dbeats.sort()

        elif extra_beat_mode == 1:

            beats = self.beats
            dbeats = []

            for beat_sec, beat_num in beats:
                if int(beat_num) in (1,3):
                    dbeats.append(beat_sec)
            
            onsets =[]
            for b1, b3 in zip(dbeats,dbeats[1:]):
                diff = (b3-b1)/4
                start = b1+diff
                end = b3-diff
                onset_unit = [o for o in self.raw_onsets if start<= o < end]
                if len(onset_unit) > onset_limit:
                    onset_unit = onset_sample(onset_unit, onset_limit, start, end)
                onsets.append( onset_unit )

            onsets = list(chain.from_iterable(onsets))
            dbeats += onsets
            dbeats.sort()

        elif extra_beat_mode == 2:

            beats = self.beats
            dbeats = []

            for beat_sec, beat_num in beats:
                if int(beat_num) in (1,2,4):
                    dbeats.append(beat_sec)
            
            b2_list = [b[0] for b in beats if int(b[1]) == 2]
            b4_list = [b[0] for b in beats if int(b[1]) == 4]
            b4_list = [b for b in b4_list if b > b2_list[0]]

            onsets =[]
            for b2, b4 in zip(b2_list, b4_list):
                diff = (b4-b2)/4
                start = b2+diff
                end = b4-diff
                onset_unit = [o for o in self.raw_onsets if start<= o < end]
                if len(onset_unit) > onset_limit:
                    onset_unit = onset_sample(onset_unit, onset_limit, start, end)
                onsets.append( onset_unit )

            onsets = list(chain.from_iterable(onsets))
            dbeats += onsets
            dbeats.sort()
        
        elif extra_beat_mode == 3:

            beats = self.beats
            dbeats = []

            for beat_sec, beat_num in beats:
                if int(beat_num) in (1,2,3,4):
                    dbeats.append(beat_sec)
            
            b3_list = [b[0] for b in beats if int(b[1]) == 3]
            b4_list = [b[0] for b in beats if int(b[1]) == 4]
            b4_list = [b for b in b4_list if b > b3_list[0]]

            extra =[]
            for b3, b4 in zip(b3_list, b4_list):
                extra.append( (b3+b4)/2 )

            dbeats += extra
            dbeats.sort()

        elif extra_beat_mode == 4:

            beats = self.beats
            dbeats = []

            for beat_sec, beat_num in beats:
                if int(beat_num) in (1,2,3,4):
                    dbeats.append(beat_sec)
            
            b2_list = [b[0] for b in beats if int(b[1]) == 2]
            b3_list = [b[0] for b in beats if int(b[1]) == 3]
            b4_list = [b[0] for b in beats if int(b[1]) == 4]
            b3_list = [b for b in b3_list if b > b2_list[0]]
            b4_list = [b for b in b4_list if b > b2_list[0]]

            extra =[]
            for i,(b2, b3, b4) in enumerate(zip(b2_list, b3_list, b4_list)):
                if i%2 == 0:
                    extra.append( (b3+b4)/2 )
                else:
                    extra.append( (b2+b3)/2 )

            dbeats += extra
            dbeats.sort()

        dbeats_time_to_audio_index = np.array(dbeats, dtype=float) * self.sample_rate
        self.onsets = np.unique(np.array(dbeats_time_to_audio_index, dtype=int))

        logger.info(f"onset end {time.time() - start_tim}")

        return self.onsets

    def load_onsets(self):
        annotation_beats_path = Song.path_to_annotation_file("music_swap_annotations_onsets", self.song_name)

        if os.path.exists(annotation_beats_path):
            self.raw_onsets = np.loadtxt(annotation_beats_path)
        else:
            # there is no beats annotation
            self.annotate_onsets(annotation_beats_path)
            # log here
            self.load_onsets()

    def get_beats44(self):
        if self.beats44 is not None:
            return self.beats44

        beats = self.beats
        dbeats = []
        for beat_sec, beat_num in beats:
            dbeats.append(beat_sec)
        dbeats_time_to_audio_index = np.array(dbeats, dtype=float) * self.sample_rate
        self.beats44 = np.array(dbeats_time_to_audio_index, dtype=int)
        return self.beats44


def clear_cache():
    import shutil

    if os.path.exists('music_swap_annotations_onsets'):
        shutil.rmtree('music_swap_annotations_onsets')
    if os.path.exists('music_swap_annotations'):
        shutil.rmtree('music_swap_annotations')

def create_all_dirs():
    dir_list = [
        "music_swap_annotations_onsets",
        "music_swap_annotations",
        "output",
        "tmp",
    ]
    for d in dir_list:
        Path(d).mkdir(exist_ok=True)


def time_stretch(audio, factor, sample_rate=44100):
    import pyrubberband as pyrb
    return pyrb.time_stretch(audio, sample_rate, factor)

def update_key(d:dict, target_k:str, f):
    def update_key_list(l:list, target_k:str, f):
        for item in l:
            if type(item) == dict:
                update_key( item, target_k, f)
            elif type(item) == list:
                update_key_list( item, target_k, f)

    for k,v in d.items():
        if k == target_k:
            d[k] = f(v)
        elif type(v) == dict:
            update_key( v, target_k, f )
        elif type(v) == list:
            update_key_list( v, target_k, f )

def to_seconds(v):
    def convert(timestr):
        logger.debug(f"{timestr=}")
        seconds= 0
        for part in timestr.split(':'):
            seconds= seconds*60 + float(part)
        return seconds

    if type(v) == str:
        return convert(v)
    elif type(v) == list:
        return [ to_seconds(i) for i in v]
    else:
        return v


def encode_video(video, out_path, opt, ffmpeg_params = []):
    codec = opt["codec"]
    preset = opt["preset"]

    logger.debug(f"{opt=}")

    if opt.get("crf") != None:
        ffmpeg_params += ["-crf", str(opt.get("crf"))]

    if codec == "rawvideo":
        ffmpeg_params += ["-pix_fmt","bgr24"]
        out_path = str(Path(out_path).with_suffix(".avi"))
    
    logger.info(f"{codec=}")
    logger.info(f"{preset=}")
    logger.info(f"{ffmpeg_params=}")

    video.write_videofile(out_path, codec=codec, preset=preset, ffmpeg_params = ffmpeg_params)

    return out_path

def debug_dump_video(video:VideoFileClip, out_path):
    debug_opt={
        "codec":"libx264",
        "preset":"ultrafast",
        "crf": 20,
    }

    outputdir = Path("output/")
    outputdir.mkdir(exist_ok=True)

    encode_video(video, str(outputdir) + "/" + out_path, debug_opt, ["-vf","scale=-2:480"])

def get_timestr():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_outputfile_name(video_path,audio_path,suffix=".mp4"):
    from pathlib import Path

    outputdir = Path("output/")
    outputdir.mkdir(exist_ok=True)

    tmp = str(outputdir) + "/" +  get_timestr() + "_" + Path(video_path).stem + "_" +Path(audio_path).stem
    return tmp[:256] + suffix

def reverse_clip(clip:VideoClip):
    logger.debug(f"{clip.duration=}")
    clip = clip.subclip(clip.duration/2)
    rclip = vfx.time_mirror(clip)
    rclip = vfx.time_symmetrize(rclip)

    logger.debug(f"{rclip.duration=}")
    return rclip

def create_tmp_speed_audio(file_path, speed):

    class SpeedChanger:
        def __init__(self, filepath=None):
            self.filepath = filepath
            self.audio = None
            self.sample_rate = 44100

            if filepath is not None:
                self.load_song_audio()

        def load_song_audio(self):
            from essentia.standard import AudioLoader
            self.audio,self.sample_rate,numberChannels,md5,bit_rate,codec = AudioLoader(filename=self.filepath)()
            self.sample_rate = int(self.sample_rate)

        def save_song_audio(self, new_filepath):
            from essentia.standard import AudioWriter

            new_format = new_filepath.split('/')[-1].split('.')[1]
            AudioWriter(filename=new_filepath, bitrate=320, sampleRate=self.sample_rate, format=new_format)(self.audio)

        def speed_change(self, factor):
            self.audio = time_stretch(self.audio, factor, sample_rate=self.sample_rate)

    s = SpeedChanger(file_path)
    s.speed_change(speed)
    tmpstr = "tmp_spd_" + get_timestr() + ".wav"
    s.save_song_audio(tmpstr)
    return tmpstr


def combine_clips(clip_list, output_path, encode_option):

    if len(clip_list) == 1:
        final_video = clip_list[0]
    else:
        final_video = concatenate_videoclips(clip_list)

    output_path = encode_video(final_video, output_path, encode_option)
 
    return output_path


_scene_map={}
def detect_scene(file_path, start_sec, end_sec, min_scene_length, scene_detection_threshold, use_cache=True):
    global _scene_map

    if (not use_cache) or (not _scene_map.get(file_path, None)):
        from scenedetect import detect, ContentDetector, AdaptiveDetector
        scene_list = detect(file_path, ContentDetector(threshold=scene_detection_threshold, min_scene_len=min_scene_length), show_progress=True)
        #scene_list = detect(file_path, AdaptiveDetector(min_scene_len=min_scene_length), show_progress=True)
        _scene_map[file_path] = [(s[0].get_seconds(),s[1].get_seconds()) for s in scene_list]

    scenes = _scene_map[file_path]

    if end_sec != -1:
        scenes = [(s[0] - start_sec, s[1] - start_sec) for s in scenes if start_sec <= s[0] < end_sec]
        if (end_sec-start_sec) - scenes[-1][1] > 1.5:
            scenes.append( ( scenes[-1][1], end_sec-start_sec) )
    else:
        scenes = [(s[0] - start_sec, s[1] - start_sec) for s in scenes if start_sec <= s[0]]
    
    if scenes[0][0] > 1.5:
        scenes = [(0, scenes[0][0])] + scenes

    return scenes



def score_audio_rms(audio, cur_tim):
    import madmom

    def zscore(x, axis = None):
        xmean = x.mean(axis=axis, keepdims=True)
        xstd  = np.std(x, axis=axis, keepdims=True)
        zscore = (x-xmean)/xstd
        return zscore
    
    # 0.125 sec / 2
    fps = 16
    sig = madmom.audio.signal.FramedSignal(audio, fps=fps, frame_size= 44100 / float(fps) )
    rms = madmom.audio.signal.root_mean_square(sig)
    rms = zscore(rms)
    logger.debug(rms)

    if False:
        rms_diff = np.pad( np.diff(rms), (1,0), mode='constant', constant_values=0)
        rms_diff = np.pad( rms_diff, (0, 16 - (len(rms_diff) % 16) ), mode='constant', constant_values=0)
        rms_diff = np.reshape(rms_diff, (-1,16))
        rms_diff = (rms_diff*100).astype(np.int32)
        rms_diff = abs(rms_diff)
        logger.info(f"{np.average(rms_diff)=}")
        rms_diff[ rms_diff < np.median(rms_diff)] = 0
        rms_diff = np.sum(rms_diff, axis=1)

        tmpstr = "RMSDIFF_" + cur_tim + "_" + Path(audio).stem + ".json"
        np.savetxt(tmpstr, rms_diff, fmt="%.0f", newline="\n")

        logger.info(f"{tmpstr=}")
        analyze_rms(rms_diff - np.median(rms_diff))

    
    rms = np.pad( rms, (0, 16 - (len(rms) % 16) ), mode='constant', constant_values=-1.5)
    rms = np.reshape(rms, (-1,16))
    rms = np.sum(rms, axis=1)
    logger.debug(rms)

    for i,r in enumerate(rms):
        logger.debug(f"{i} sec {r}")

    if False:
        tmpstr = "RMS_" + cur_tim + "_" + Path(audio).stem + ".json"
        np.savetxt(tmpstr, (np.array(rms) ).astype(np.int32), fmt="%.0f", newline="\n")

    return rms

def analyze_rms(rms, boundary_hint):

    boundary=[]

    prev_flag = -1
    cur_flag = -1
    # -1 ... init
    # 0 ... plus
    # 1 ... minus
    # 2 ... same as prev

    for i in range(len(rms)-5):
        cur = rms[i:i+6]
        logger.debug(f"{cur=}")
        cur = [c > 0 for c in cur]
        if sum(cur) > 4:
            cur_flag = 0
        elif sum( [not a for a in cur] ) > 4:
            cur_flag = 1
        else:
            cur_flag = 2
        
        if prev_flag == -1:
            boundary.append( (i, cur_flag) )
        elif cur_flag in [0,1]:
            if cur_flag != prev_flag:
                if boundary[0][1] == 2:
                    boundary[0] = (0, cur_flag)
                else:
                    boundary.append( (i, cur_flag) )
        elif cur_flag == 2:
            cur_flag = prev_flag

        prev_flag = cur_flag
    
    logger.debug(f"{boundary=}")

    if boundary_hint is None:
        return boundary

    boundary2=[]
    for b in boundary:
        boundary_hint = boundary_hint[b[0] <= boundary_hint]
        if len(boundary_hint) > 0:
            boundary2.append( ( boundary_hint[0], b[1] ) )
        else:
            logger.info(f"{len(boundary_hint)=}")
            boundary2.append( b )
    
    boundary2[0] = (0, boundary2[0][1])

    logger.info(f"{boundary2=}")

    return boundary2
    


def analyze_music_boundary(audio_path, boundary_hint):
    from pydub import AudioSegment

    start_tim = time.time()

    cur_tim = get_timestr()

    rms1 = score_audio_rms(audio_path, cur_tim)

    logger.debug(f"rms1 : {time.time() - start_tim}")        

    tmpstr = cur_tim + "_LPF50_" + Path(audio_path).stem + ".wav"

    song = AudioSegment.from_file(audio_path)
    bass = song.low_pass_filter(50)
    bass.export(tmpstr)

    rms2 = score_audio_rms(tmpstr, cur_tim)

    os.remove(tmpstr)

    logger.debug(f"rms2 : {time.time() - start_tim}")        

    rms_sum = rms1 + rms2

    #np.savetxt( cur_tim + "_sum.json" , rms_sum  , fmt="%.02f", newline="\n")

    boundary = analyze_rms(rms_sum, boundary_hint)

    logger.info(f"analyze_music_boundary : {time.time() - start_tim}")

    return boundary

def get_audio_length(audio_file_path):
    from pydub import AudioSegment
    song = AudioSegment.from_file(audio_file_path)
    return len(song)/1000


def ogg_to_wav(file_path):
    audio_path = Path(file_path)
    if audio_path.suffix in (".ogg",".opus"):
        wav_path = audio_path.with_suffix(".wav")
        if not wav_path.exists():
            from pydub import AudioSegment
            seg = AudioSegment.from_ogg(file_path)
            seg.export(wav_path, format="wav")
        file_path = str(wav_path)
    return file_path

