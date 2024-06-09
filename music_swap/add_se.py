import random
import logging
from pathlib import Path

from moviepy.editor import *
import moviepy.video.fx.all as vfx


from music_swap.util import *



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



def create_sound_effect_list(beats, info):
    import madmom

    onset_dict={}
    onset_proc = madmom.features.OnsetPeakPickingProcessor(fps=100)
    onset_proc2 = madmom.features.RNNOnsetProcessor()

    def get_wav_onset(wav_file_path):
        nonlocal onset_dict
        if wav_file_path in onset_dict:
            onset = onset_dict[wav_file_path]
            return onset[0] if len(onset) > 0 else 0
        activations = onset_proc2(wav_file_path)
        wav_onsets = onset_proc(activations)
        logger.debug(f"{wav_onsets=}")
        onset_dict[wav_file_path] = wav_onsets
        return wav_onsets[0] if len(wav_onsets) > 0 else 0
    
    wav_list=[]
    wavfile = Path( info["wav_file_or_dir_path"] )
    if wavfile.is_dir():
        wav_list = list(wavfile.glob("**/*.wav")) + list(wavfile.glob("**/*.mp3"))
    else:
        wav_list = [wavfile]


    def pick_wav():
        nonlocal wav_list
        random.shuffle(wav_list)

        if len(wav_list) == 0:
            return None

        if info["once_per_file"]:
            return str(wav_list.pop())
        else:
            return str(wav_list[0])

    audio_list = []

    if len(beats) == 1:
        wav = pick_wav()
        logger.debug(f"{wav=}")
        
        wav_onset = get_wav_onset(wav)
        logger.debug(f"{wav_onset=}")
        audio = AudioFileClip(wav)
        audio = audio.set_start(beats[0] - wav_onset)
        audio_list.append(audio)

    clip_map = {}
    def get_clip(file_path):
        nonlocal clip_map
        if not clip_map.get(file_path, None):
            clip_map[file_path] = AudioFileClip(file_path)
        return clip_map[file_path]
    

    for i,(b0,b1) in enumerate(zip(beats, beats[1:])):
        logger.debug(f"{b0=}")
        logger.debug(f"{(b1-b0)=}")

        wav = pick_wav()
        logger.debug(f"{wav=}")

        if wav == None:
            break
        
        wav_onset = get_wav_onset(wav)
        logger.debug(f"{wav_onset=}")
        audio = get_clip(wav)
        limit_len = (b1-b0)*2 + wav_onset
        if limit_len < audio.duration:
            audio = audio.set_duration( limit_len )
        audio = audio.set_start(b0 - wav_onset)
        audio_list.append(audio)

    audio_list = [a.volumex( info["volume"] ) for a in audio_list]

    return audio_list


###############################################################

def exec_se_command(video, se_info_list):

    class Beats:
        def __init__(self, filepath):
            self.filepath = filepath
            self.onsets = None
            self.beats = None
            self.beats44 = None
            self.beats24 = None
            self.beats14 = None

            if filepath is not None:
                self.load_beats()
                self.load_onsets()
        
        def load_beats(self):
            import madmom
            downbeats_proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
            activations = madmom.features.RNNDownBeatProcessor()(self.filepath)
            self.beats = downbeats_proc(activations)
            self.beats44 = [b[0] for b in self.beats if b[1] in [1,2,3,4]]
            self.beats24 = [b[0] for b in self.beats if b[1] in [1,3]]
            self.beats14 = [b[0] for b in self.beats if b[1] in [1]]

        def load_onsets(self):
            if False:
                from pydub import AudioSegment
                song = AudioSegment.from_file(self.filepath)
                bass = song.low_pass_filter(50)
                tmp_file = "percussive_" + get_timestr() + ".wav"
                bass.export(tmp_file)
            if True:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(self.filepath)
                y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))
                tmp_file = "percussive_" + get_timestr() + ".wav"

                sf.write(tmp_file, y_percussive, sr, 'PCM_24')


            #downbeats_proc = madmom.features.OnsetPeakPickingProcessor(threshold=0.5, combine=0.15, fps=100)
            downbeats_proc = madmom.features.OnsetPeakPickingProcessor(combine=0.1, fps=100)
            activations = madmom.features.RNNOnsetProcessor()(tmp_file)
            #activations = madmom.features.RNNOnsetProcessor()(self.filepath)
            self.onsets = downbeats_proc(activations)

            logger.info(f"{len(self.onsets)=}")

            if False:
                mod=[]
                prev_b = self.onsets[0]
                for b0 in self.onsets:
                    if b0 - prev_b < 0.2:
                        continue
                    mod.append(b0)
                    prev_b = b0

                self.onsets = np.array(mod)

                logger.info(f"{len(self.onsets)=}")


        
        def get_beats44(self, start, end):
            return [b for b in self.beats44 if start <= b < end ]
        def get_beats24(self, start, end):
            return [b for b in self.beats24 if start <= b < end ]
        def get_beats14(self, start, end):
            return [b for b in self.beats14 if start <= b < end ]
        def get_onsets(self, start, end):
            return [b for b in self.onsets if start <= b < end ]


    beats = Beats(video)
    audio_list = []

    audio_only = Path(video).suffix in (".wav", ".mp3", ".ogg", ".m4a", ".opus")

    if audio_only:
        clip = AudioFileClip(video)
        audio_list.append(clip)
    else:
        clip = VideoFileClip(video)
        audio_list.append(clip.audio)

    for info in se_info_list:
        start_sec = info["start_sec"]
        end_sec = info["end_sec"]
        if end_sec == -1:
            end_sec = clip.duration + 1

        if info["on_onset"]:
            b = beats.get_onsets(start_sec, end_sec)
        elif info["on_every_beat"]:
            b = beats.get_beats44(start_sec, end_sec)
        elif info["on_every_two_beat"]:
            b = beats.get_beats24(start_sec, end_sec)
        else:
            b = beats.get_beats14(start_sec, end_sec)
        
        if not b:
            b = [start_sec]

        ef_list = create_sound_effect_list(b, info)

        audio_list += ef_list
    
    audio = CompositeAudioClip(audio_list)

    if audio_only:
        clip = ColorClip(size=(200,200), color=(0,0,0), duration = audio.duration)
        clip.fps=24
        clip.audio = audio
    else:
        clip.audio = audio

    return clip



