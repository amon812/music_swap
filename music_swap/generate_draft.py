import random
import logging
from pathlib import Path
import copy

from moviepy.editor import *
import moviepy.video.fx.all as vfx


from music_swap.util import *



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)




def create_boundary_list(video_file_path, scene_detection_threshold, min_scene_length, video_start, video_end):

    clip = VideoFileClip(video_file_path)

    min_scene_length = min_scene_length * clip.fps

    clip.close()

    org_video_scenes = detect_scene(video_file_path, 0, -1, min_scene_length, scene_detection_threshold)
    org_video_scenes = [s[0] for s in org_video_scenes]
    org_video_scenes.append( get_audio_length(video_file_path) )
    org_video_scenes = np.array(org_video_scenes)
    org_video_boundary = analyze_music_boundary(video_file_path, org_video_scenes)
    org_video_boundary.append((get_audio_length(video_file_path), -1))

    bg_of_video = Song2(video_file_path, True)
    video_beats = bg_of_video.get_beats44()
    video_beats = np.array([v/44100 for v in video_beats])

    def get_cut_list(st,end):
        def get_beat_count(st,end):
            return ((st <= video_beats) & (video_beats < end)).sum()
        
        if end != -1:
            cuts = [ c for c in org_video_scenes if st <= c <= end ]
        else:
            cuts = [ c for c in org_video_scenes if st <= c ]

        result = []
        for c0,c1 in zip(cuts, cuts[1:]):
            result.append( (c0, c1, get_beat_count(c0,c1)) )
            
        return result
    

    boundary = []

    pending = None

    for i,(b0,b1) in enumerate(zip(org_video_boundary, org_video_boundary[1:])):
        r = ( b0[0], b1[0] )

        if video_start > r[1]:
            continue
        elif r[0] <= video_start < r[1]:
            r = (video_start, r[1])
        
        if video_end != -1:
            if video_end < r[0]:
                continue
            elif r[0] <= video_end < r[1]:
                r = (r[0], video_end)
        
        if pending:
            logger.debug(f"before {r=}")
            r = (pending[0], r[1])
            logger.debug(f"after {r=}")
            pending = None

        cut_list = get_cut_list(r[0], r[1])
        
        r = ( r[0], r[1], cut_list )

        logger.debug(f"{r[0]=} {r[1]=}")
        logger.debug(f"{cut_list=}")

        if len(cut_list) == 0:
            pending = (r[0], r[1])

            continue

        boundary.append( (r, b0[1]) )
    
    if pending and len(boundary) > 0:
        logger.debug(f"before {boundary[-1]=}")
        boundary[-1] = ((boundary[-1][0][0], pending[1], boundary[-1][0][2]),boundary[-1][1])
        pending = None
        logger.debug(f"after {boundary[-1]=}")
    
    return [b[0] for b in boundary if b[1]==0 ], [b[0] for b in boundary if b[1]==1 ]


def create_audio_boundary(audio_file_path):
    def get_downbeats(audio_file_path):
        song = Song(audio_file_path)
        downbeats = song.get_downbeats()
        downbeats = np.array([d/44100 for d in downbeats])
        return downbeats
        
    return analyze_music_boundary(audio_file_path, get_downbeats(audio_file_path))




###############################################################
def exec_generate_draft_command(info_all, video, org_new_audio, video_start, video_end):
    random.seed()

    tmp_file_created = False

    base_speed = info_all.get("base_speed", 1.0)
    if base_speed != 1.0:
        new_audio = create_tmp_speed_audio(org_new_audio, base_speed)
        tmp_file_created = True
    else:
        new_audio = org_new_audio


    generate_draft_option = info_all.get("generate_draft_option", {})

    scene_detection_threshold = generate_draft_option.get("scene_detection_threshold", 50)
    scene_detection_min_sec = generate_draft_option.get("scene_detection_min_sec", 2.0)
    onset_for_loud_area = generate_draft_option.get("stretch_method_1_for_loud_area", False)
    random_scenes_mode = generate_draft_option.get("random_scenes", False)
    random_shots_mode = generate_draft_option.get("random_shots", False)
    story_flag = generate_draft_option.get("follow_the_original_scene_order", False)
    story_flag_no_skipping = generate_draft_option.get("no_skipping_and_follow_the_original_scene_order", False)
    no_reuse_mode = generate_draft_option.get("no_reuse_of_scenes", False)
    last_cut_mode = generate_draft_option.get("last_shot_mode", False)
    last_cut_length = generate_draft_option.get("last_shot_sec", 8.0)

    story_mode = story_flag or story_flag_no_skipping

    logger.info(f"{scene_detection_threshold=}")
    logger.info(f"{scene_detection_min_sec=}")
    logger.info(f"{onset_for_loud_area=}")
    logger.info(f"{random_scenes_mode=}")
    logger.info(f"{random_shots_mode=}")
    logger.info(f"{story_flag=}")
    logger.info(f"{story_flag_no_skipping=}")
    logger.info(f"{no_reuse_mode=}")
    logger.info(f"{last_cut_mode=}")
    logger.info(f"{last_cut_length=}")

    if story_mode:
        no_reuse_mode = True
        random_scenes_mode = False
        last_cut_mode = True


    m0_boundary, m1_boundary = create_boundary_list( video, scene_detection_threshold, scene_detection_min_sec, video_start, video_end )

    if len(m0_boundary) == 0:
        m0_boundary=m1_boundary
    elif len(m1_boundary) == 0:
        m1_boundary=m0_boundary

    audio_boundary = create_audio_boundary(new_audio)

    ### audio - file_path
    info_all["audio"]["file_path"] = org_new_audio
    ### audio - boundary
    info_all["audio"]["boundary"] = [a[0] for a in audio_boundary]

    audio_boundary.append((get_audio_length(new_audio), -1))



    base_unit = info_all["video"][0][0]
    base_unit["file_path"] = video
    base_unit["option"]["scene_detection_threshold"] = scene_detection_threshold
    base_unit["option"]["scene_detection_min_sec"] = scene_detection_min_sec

    video_scene_list = []

    audio_song = Song2(new_audio)
    audio_beats44 = audio_song.get_beats44()
    audio_beats44 = np.array([d/44100 for d in audio_beats44])
    audio_onsets = audio_song.get_onsets()
    audio_onsets = np.array([d/44100 for d in audio_onsets])

    video_song = Song2(video)
    video_beats44 = video_song.get_beats44()
    video_beats44 = np.array([d/44100 for d in video_beats44])

    def get_beat_count(st, end, is_onset):
        if is_onset:
            return ((st <= audio_onsets) & (audio_onsets < end)).sum()
        else:
            return ((st <= audio_beats44) & (audio_beats44 < end)).sum()
        
    def get_video_beat_count(st, end):
        return ((st <= video_beats44) & (video_beats44 < end)).sum()

    
    ##################################################
    class Resource:
        # r -> ( start_sec, end_sec, cut_list )
        # cut_list -> [ (start_sec, end_sec, beat_count), (start_sec, end_sec, beat_count), ... ]

        def __init__(self, m0_list, m1_list, no_rewind):
            self.org_m0_list = m0_list.copy()
            self.org_m1_list = m1_list.copy()
            self.m0_list = m0_list
            self.m1_list = m1_list
            self.m0_list_index = -1
            self.m0_cut_index = -1
            self.m1_list_index = -1
            self.m1_cut_index = -1
            self.no_rewind = no_rewind

            if no_rewind:
                new_m0_list = m0_list + m1_list
                logger.debug(f"before {new_m0_list=}")
                new_m0_list = sorted(new_m0_list, key=lambda x: x[0] )
                logger.debug(f"after {new_m0_list=}")
                self.m0_list = new_m0_list
                self.m1_list = []

            self._clear()
        
        def shuffle(self):
            random.shuffle( self.m0_list )
            random.shuffle( self.m1_list )
        
        def _clear(self):
            if len(self.m0_list) > 0:
                self.m0_list_index = 0
                self.m0_cut_index = 0
            if len(self.m1_list) > 0:
                self.m1_list_index = 0
                self.m1_cut_index = 0
        
        def _inc(self, is_m0):
            carry = False
            if is_m0:
                self.m0_cut_index += 1
                if len(self.m0_list[ self.m0_list_index ][2] ) == self.m0_cut_index:
                    self.m0_list_index += 1
                    self.m0_cut_index = 0
                    carry = True
                    if len( self.m0_list ) == self.m0_list_index:
                        self.m0_list_index = -1
                        self.m0_cut_index = -1
            else:
                self.m1_cut_index += 1
                if len(self.m1_list[ self.m1_list_index ][2] ) == self.m1_cut_index:
                    self.m1_list_index += 1
                    self.m1_cut_index = 0
                    carry = True
                    if len( self.m1_list ) == self.m1_list_index:
                        self.m1_list_index = -1
                        self.m1_cut_index = -1
            
            return carry

        def _is_empty(self, is_m0):
            if is_m0:
                return self.m0_list_index == -1
            else:
                return self.m1_list_index == -1
        
        def is_empty(self):
            return self._is_empty(True) and self._is_empty(False)

        def pop(self, is_m0):
            if self.is_empty():
                self._clear()

            if self._is_empty(is_m0):
                is_m0 = not is_m0
            
            if is_m0:
                c = self.m0_list[ self.m0_list_index ][2][self.m0_cut_index]
            else:
                c = self.m1_list[ self.m1_list_index ][2][self.m1_cut_index]

            is_carry = self._inc(is_m0)

            return c, is_carry
        
        def _get_cut_end(self, start, beat_count):
            cut_time = video_beats44[ start <= video_beats44 ][:beat_count]
            return cut_time[-1]
        
        def _dump(self, comment):
            logger.debug(comment)
            logger.debug(f"{len(self.m0_list)=}")
            logger.debug(f"{self.m0_list_index=}")
            logger.debug(f"{self.m0_cut_index=}")
            logger.debug(f"{len(self.m1_list)=}")
            logger.debug(f"{self.m1_list_index=}")
            logger.debug(f"{self.m1_cut_index=}")
        
        def forward_to(self, is_m0):
            if not self.no_rewind:
                return
            
            logger.debug(f"forward_to before {self.m0_list_index=}")
            
            cur_tim = self.m0_list[ self.m0_list_index ][0]

            current_is_m0 = False

            for m in self.org_m0_list:
                if cur_tim == m[0]:
                    current_is_m0 = True
                    break
            
            if is_m0 == current_is_m0:
                return
            else:
                target_list = self.org_m0_list if is_m0 else self.org_m1_list
                target_list = [m[0] for m in target_list if m[0] > cur_tim]

                for i in range(self.m0_list_index, len(self.m0_list)):
                    if self.m0_list[i][0] in target_list:
                        self.m0_list_index = i
                        self.m0_cut_index = 0
                        break

            logger.debug(f"forward_to after {self.m0_list_index=}")
        
        def pop2(self, beat_count, is_m0):
            self._dump("== before == ")
            if self.is_empty():
                self._clear()

            if self._is_empty(is_m0):
                is_m0 = not is_m0
            
            if is_m0:
                cut_list = self.m0_list[ self.m0_list_index ][2]
                start = self.m0_list[ self.m0_list_index ][2][self.m0_cut_index][0]
                end = self.m0_list[ self.m0_list_index ][2][-1][1]
            else:
                cut_list = self.m1_list[ self.m1_list_index ][2]
                start = self.m1_list[ self.m1_list_index ][2][self.m1_cut_index][0]
                end = self.m1_list[ self.m1_list_index ][2][-1][1]
            
            remain = get_video_beat_count( start, end )
            if remain <= beat_count:
                is_carry = True
                beat_count = remain
            else:
                is_carry = False
            
            if is_carry:
                if is_m0:
                    self.m0_list_index += 1
                    self.m0_cut_index = 0
                    if len( self.m0_list ) == self.m0_list_index:
                        self.m0_list_index = -1
                        self.m0_cut_index = -1
                else:
                    self.m1_list_index += 1
                    self.m1_cut_index = 0
                    if len( self.m1_list ) == self.m1_list_index:
                        self.m1_list_index = -1
                        self.m1_cut_index = -1
            else:
                raw_end = self._get_cut_end(start, beat_count)
                logger.debug(f"{raw_end=}")
                logger.debug(f"{cut_list=}")
                new_index = next(i for i,v in enumerate(cut_list) if v[1] >= raw_end )
                end = cut_list[new_index][1]
                if is_m0:
                    self.m0_cut_index = new_index
                else:
                    self.m1_cut_index = new_index
                self._inc(is_m0)
                
            self._dump("== after == ")
            return (start, end, beat_count)
    ##################################################

    for m in m0_boundary:
        logger.debug(f"{m=}")
    for m in m1_boundary:
        logger.debug(f"{m=}")

    res = Resource(m0_boundary, m1_boundary, story_mode)

    if random_scenes_mode:
        res.shuffle()

    for i,(b0,b1) in enumerate(zip(audio_boundary, audio_boundary[1:])):
        cur_length = b1[0]-b0[0]
        logger.info(f"{i} : {cur_length=}")
        logger.info(f"{b0[1]=}")

        video_scene = []

        ##################################################
        def story_mode_func(index, start_pos, end_pos, is_loud_area):
            nonlocal video_scene, res

            is_onset = False
            if onset_for_loud_area:
                is_onset = is_loud_area

            cur_beat_count = get_beat_count(start_pos, end_pos, is_onset)
            logger.info(f"{cur_beat_count=}")

            cur_beat_count += 2       #margin

            cur_cut = None
            cut_list = []

            if not story_flag_no_skipping:
                if is_loud_area:
                    res.forward_to( True )
            
            res_is_empty = False

            while(cur_beat_count > 0):
                cut = res.pop2( cur_beat_count, is_loud_area)
                logger.debug(f"res.pop2 result {cut=}")
                cur_beat_count -= cut[2]
                if cur_cut:
                    cur_cut = (cur_cut[0], cut[1])
                else:
                    cur_cut = cut
                
                cut_list.append(cut)

                if no_reuse_mode:
                    if res.is_empty():
                        res_is_empty = True
                        break

            v_start_sec = cur_cut[0]
            v_end_sec = cur_cut[1]

            unit = copy.deepcopy( base_unit )
            unit["start_sec"] = v_start_sec
            unit["end_sec"] = v_end_sec

            unit["option"]["random_shots"] = False
            unit["option"]["stretch_method"] = 1 if is_onset else 0
            unit["option"]["use_once"] = True
            unit["option"]["fadeout"] = 0
            
            video_scene.append(unit)

            if res_is_empty:
                return
            
            if last_cut_mode:
                if len(cut_list) > 1:
                    unit = copy.deepcopy( base_unit )
                    unit["start_sec"] = cut_list[-1][0]
                    unit["end_sec"] = v_end_sec

                    unit["option"]["random_shots"] = False
                    unit["option"]["stretch_method"] = 1 if is_onset else 0
                    unit["option"]["use_once"] = True
                    unit["option"]["fadein"] = 0.2
                    unit["option"]["fadeout"] = 0
                    unit["option"]["fill_from_back"] = True
                    
                    video_scene.append(unit)
                else:
                    if v_end_sec > last_cut_length:
                        unit = copy.deepcopy( base_unit )
                        unit["start_sec"] = v_end_sec - last_cut_length
                        unit["end_sec"] = v_end_sec

                        unit["option"]["random_shots"] = False
                        unit["option"]["stretch_method"] = 1 if is_onset else 0
                        unit["option"]["use_once"] = True
                        unit["option"]["fadein"] = 0.2
                        unit["option"]["fadeout"] = 0
                        unit["option"]["fill_from_back"] = True
                        
                        video_scene.append(unit)

        ##################################################
        def random_mode_func(index, start_pos, end_pos, is_loud_area):
            nonlocal video_scene, res

            is_onset = False
            if onset_for_loud_area:
                is_onset = is_loud_area

            cur_beat_count = get_beat_count(start_pos, end_pos, is_onset)
            logger.info(f"{cur_beat_count=}")

            cur_beat_count *= 1.25       #margin
            cur_beat_count = int(cur_beat_count)

            while(cur_beat_count > 0):
                
                cut = res.pop2( cur_beat_count, is_loud_area)
                logger.debug(f"res.pop2 result {cut=}")
                v_start_sec = cut[0]
                v_end_sec = cut[1]

                cur_beat_count -= cut[2]

                unit = copy.deepcopy( base_unit )
                unit["start_sec"] = v_start_sec
                unit["end_sec"] = v_end_sec

                unit["option"]["random_shots"] = True
                unit["option"]["stretch_method"] = 1 if is_onset else 0
                unit["option"]["use_once"] = False

                video_scene.append(unit)

                if no_reuse_mode:
                    if res.is_empty():
                        break

            video_scene[-1]["option"]["fadeout"] = 0.2


        ##################################################
        def normal_mode_func(index, start_pos, end_pos, is_loud_area):
            nonlocal video_scene, res

            is_onset = False
            if onset_for_loud_area:
                is_onset = is_loud_area

            cur_beat_count = get_beat_count(start_pos, end_pos, is_onset)
            logger.info(f"{cur_beat_count=}")

            cur_beat_count += 2       #margin

            cut_list = []
            
            res_is_empty = False

            while(cur_beat_count > 0):
                cut = res.pop2( cur_beat_count, is_loud_area)
                logger.debug(f"res.pop2 result {cut=}")
                cur_beat_count -= cut[2]
                
                cut_list.append(cut)

                if no_reuse_mode:
                    if res.is_empty():
                        res_is_empty = True
                        break
            
            
            for cut in cut_list:
                v_start_sec = cut[0]
                v_end_sec = cut[1]

                unit = copy.deepcopy( base_unit )
                unit["start_sec"] = v_start_sec
                unit["end_sec"] = v_end_sec

                unit["option"]["random_shots"] = False
                unit["option"]["stretch_method"] = 1 if is_onset else 0
                unit["option"]["use_once"] = True
                unit["option"]["fadeout"] = 0
                
                video_scene.append(unit)

                if not res_is_empty and last_cut_mode:
                    if len(cut_list) > 1:
                        video_scene[-1]["option"]["fadein"] = 0.2
                        video_scene[-1]["option"]["fill_from_back"] = True
                    else:

                        if v_end_sec > last_cut_length:
                            unit = copy.deepcopy( base_unit )
                            unit["start_sec"] = v_end_sec - last_cut_length
                            unit["end_sec"] = v_end_sec

                            unit["option"]["random_shots"] = False
                            unit["option"]["stretch_method"] = 1 if is_onset else 0
                            unit["option"]["use_once"] = True
                            unit["option"]["fadein"] = 0.2
                            unit["option"]["fadeout"] = 0
                            unit["option"]["fill_from_back"] = True
                            
                            video_scene.append(unit)
        ##################################################


        if story_mode:
            story_mode_func(i, b0[0], b1[0], b0[1]==0)
        elif random_shots_mode:
            random_mode_func(i, b0[0], b1[0], b0[1]==0)
        else:
            normal_mode_func(i, b0[0], b1[0], b0[1]==0)

        video_scene_list.append(video_scene)

        if no_reuse_mode:
            if res.is_empty():
                break

    video_scene_list[-1][-1]["option"]["fadeout"] = 3


    ### video
    info_all["video"] = video_scene_list

    if tmp_file_created:
        os.remove(new_audio)

    return info_all




