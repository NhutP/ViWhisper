import sys
sys.path.insert(0, r'..')

import customalign_word as al_word
import customalign_char as al_char
import pathlib
import tqdm
import argparse
from utils.prepare_data import list_leaf_dirs

space = '--------------------------------------------\n'
class BasicAligner():
    def __init__(self, match=1, mis_match=-1, gap_penalty=-2, level='word'):
        # aligner properties
        self.match = match
        self.gap_penalty = gap_penalty
        self.mis_match = mis_match
        self.level = level

        if level == 'word':
            self.scoring_obj = al_word.IdentityScoringMatrix(self.match, self.mis_match)
            self.aligner = al_word.LocalAlignment(self.scoring_obj, self.gap_penalty)
            
        elif level == 'char':
            self.scoring_obj = al_char.IdentityScoringMatrix(self.match, self.mis_match)
            self.aligner = al_char.LocalAlignment(self.scoring_obj, self.gap_penalty)
            
    def align_line(self, target_text, query_text):
        return self.aligner.align(target_text, query_text)
    


class FileAligner(BasicAligner):
    def __init__(self, accepted_threshold, pending_threshold, match=1, mis_match=-1, gap_penalty=-2, level='word', skip_query_threshold=3, cut_threshold=5, starting_search_space=2000, cumulate=200, terminate_threshold=60):
        super().__init__(match, mis_match, gap_penalty, level)

        # threshold for
        self.accepted_threshold = accepted_threshold
        self.pending_threshold = pending_threshold
        self.get_name = lambda index : f"{(index):09n}"
        self.skip_query_threshold = skip_query_threshold
        self.cut_threshold = cut_threshold
        self.starting_search_space = starting_search_space
        self.cumulate = cumulate
        self.terminate_threshold  = terminate_threshold


    def align_file(self, target_file_path, query_path :pathlib.Path):
        num_of_query = 0
        
        query_files =sorted(list(query_path.iterdir()))
        num_of_query = len(query_files)

        with open(target_file_path, 'r', encoding='utf8') as read:
            if self.level == 'word':
                target_sentence = read.read().split()
            elif self.level == 'char':
                target_sentence = read.read().strip()

        # contain index of sentences
        perfect_match_sentences_index = {}

        # contain index and match result of accepted_sentences
        accepted_sentences_index = {}

        # contain index and match result of pending sentences
        pending_sentences_index = {}

        start_index = -1
        search_space = self.starting_search_space
        consecutive_fail = 0

        get_query = lambda x : open(query_files[x], 'r', encoding='utf8').read()
        get_source_file = lambda x : query_files[x].name

        # standarlize the query
        if self.level == 'char':
            get_standard_query = lambda x : x.strip()
        elif self.level == 'word':
            get_standard_query = lambda x : x.split() 

        is_match_first = False

        for query_sentence_index in tqdm.tqdm(range(num_of_query)):
            # if fail too much, break it
            if consecutive_fail > self.terminate_threshold and not is_match_first:
                print('Consecutive fail reachs terminate threshold')
                break
            
            # get the query in standard form
            query_sentence = get_standard_query(get_query(query_sentence_index))

            query_len = len(query_sentence)

            # if too short, skip
            if  query_len < self.skip_query_threshold:
                search_space += self.cumulate
                continue

            # align a sentence to the target file
            # first argument: target, second: query
            target = target_sentence[start_index : start_index + search_space] if start_index != -1 else target_sentence

            alingment_result = self.align_line(target, query_sentence)

            # perfect match
            if alingment_result.overall_accuracy == 1:
                consecutive_fail = 0
                perfect_match_sentences_index[get_source_file(query_sentence_index)] = alingment_result.result_ref
                
                if query_len > self.cut_threshold:
                    start_index += alingment_result.r_end
                    search_space = self.starting_search_space
                    is_match_first = True
                else:
                    search_space += self.cumulate
   
                continue
                    
            # if confident, add to the accepted list
            if alingment_result.overall_accuracy >= self.accepted_threshold:
                consecutive_fail = 0
                if ('-' in alingment_result.result_ref and alingment_result.mismatches <= (query_len / 10) + 1):
                    accepted_sentences_index[get_source_file(query_sentence_index)] = alingment_result.result_query
                elif ('-' not in alingment_result.result_ref):
                    accepted_sentences_index[get_source_file(query_sentence_index)] = alingment_result.result_ref

                if query_len > self.cut_threshold:
                    start_index += alingment_result.r_end
                    search_space = self.starting_search_space
                    is_match_first = True
                    continue
                
                search_space += self.cumulate
                
                continue

            # pending match
            if alingment_result.overall_accuracy >= self.pending_threshold:
                consecutive_fail = 0
                if ('-' in alingment_result.result_ref and alingment_result.mismatches <= (query_len / 10) + 1):
                    pending_sentences_index[get_source_file(query_sentence_index)] = alingment_result.result_query
                elif ('-' not in alingment_result.result_ref):
                    pending_sentences_index[get_source_file(query_sentence_index)] = alingment_result.result_ref

                if query_len > 2 * self.cut_threshold:
                    start_index += alingment_result.r_end
                    search_space = self.starting_search_space
                    is_match_first = True
                    continue

                search_space += self.cumulate

                continue

            consecutive_fail += 1
            search_space += self.cumulate

        return perfect_match_sentences_index, accepted_sentences_index, pending_sentences_index


    def save_aligned_raw_transcript(self, output_folder :pathlib.Path, text_by_index, classifi_gap=True):
        '''
        Save specific lines of a transcript
        '''
    
        output_folder = pathlib.Path(output_folder)

        if classifi_gap:
            output_folder_gap = output_folder / 'gap'
            output_folder_nogap = output_folder / 'nogap'

            output_folder_gap.mkdir()
            output_folder_nogap.mkdir()

            get_store_folder = lambda sen, source_file_name : output_folder/ ('gap' if '-' in sen else 'nogap') / source_file_name
        
        else:
            get_store_folder = lambda sen, source_file_name : output_folder / source_file_name


        for i, sen in text_by_index.items():
            output_file = get_store_folder(sen, i)

            with open(output_file, 'w', encoding='utf8') as writer:
                writer.write(sen)


        
class FolderAligner(FileAligner):
    def __init__(self, target_folder_path, query_folder_path, result_foler_path, accepted_threshold=0.95, pending_threshold=0.9, match=1, mis_match=-1, gap_penalty=-2, level='word', skip_query_threshold=3, cut_threshold=5, starting_search_space=2000, cumulate=200, terminate_threshold=60):
        super().__init__(accepted_threshold, pending_threshold, match, mis_match, gap_penalty, level, skip_query_threshold, cut_threshold, starting_search_space, cumulate, terminate_threshold)
 
        # load paths
        # path to target pdf folder
        self.target_folder_path = pathlib.Path(target_folder_path)
        # path to query pdf (transcripted) folder
        self.query_folder_path = pathlib.Path(query_folder_path)
        # path to store the result
        self.result_foler_path = pathlib.Path(result_foler_path)


    def align_folder(self):
        leaf_query_dirs = list_leaf_dirs(self.query_folder_path)

        for query_dir in leaf_query_dirs:
            # get target file
            target_file_path = pathlib.Path(str(query_dir).replace(str(self.query_folder_path), str(self.target_folder_path)) + '.txt')
            result_dir = pathlib.Path(str(query_dir).replace(str(self.query_folder_path), str(self.result_foler_path)))

            if not target_file_path.exists():
                print(f"{target_file_path} not exists\n-----------------------------------------")
                continue

            if result_dir.exists():
                print(f'Already aligned {target_file_path} and store at {result_dir}\n------------------------------------------')
                continue

            if not result_dir.exists():
                try:
                    result_dir.mkdir(parents=True)
                    print(f'created {result_dir}')
                except Exception as e:
                    print(f'Another process may have created this folder')
                    continue

            print(f"Aligning {target_file_path}, {query_dir} and store at {result_dir}")
            # align each sentences in file with the target
            perfect_match_chunks, accpeted_match_chunks, pending_match_chunks = self.align_file(target_file_path, query_dir)

            temp_result_folder_perfect = result_dir / (result_dir.stem + '_' + '1')
            temp_result_folder_accepted = result_dir / (result_dir.stem +'_' + str(self.accepted_threshold))
            temp_result_folder_pending = result_dir / (result_dir.stem +'_' + str(self.pending_threshold))

            if not temp_result_folder_perfect.exists():
                temp_result_folder_perfect.mkdir()
                print(f'created {temp_result_folder_perfect}')
            
            if not temp_result_folder_accepted.exists():
                temp_result_folder_accepted.mkdir()
                print(f'created {temp_result_folder_accepted}')
            
            if not temp_result_folder_pending.exists():
                temp_result_folder_pending.mkdir()
                print(f'created {temp_result_folder_pending}')

            # save
            self.save_aligned_raw_transcript(temp_result_folder_perfect, perfect_match_chunks, False)
            self.save_aligned_raw_transcript(temp_result_folder_accepted, accpeted_match_chunks, True)
            self.save_aligned_raw_transcript(temp_result_folder_pending, pending_match_chunks, True)

            print(space)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('tar', help='path to target folder (formated pdf)', type=str)
    parser.add_argument('qf', help='query transcript (raw) or (chunked folder)', type=str)
    parser.add_argument('rs', help='result folder', type=str)

    parser.add_argument('--act', help='accepted threshold', type=float, default=0.95)
    parser.add_argument('--pdt', help='pending threshold', type=float, default=0.8)

    parser.add_argument('--lv', help='word level or character level', type=str, default='word')

    parser.add_argument('--mat', help='match score', default=1, type=int)
    parser.add_argument('--mma', help='mis match score', default=-1, type=int)
    parser.add_argument('--gp', help='gap penalty', default=-2, type=int)

    parser.add_argument('--sq', help='skip query', default=3, type=int)
    parser.add_argument('--cuts', help='cut threshold', default=5, type=int)
    parser.add_argument('--sp', help='starting search space', default=1000, type=int)
    parser.add_argument('--cml', help='cumulate', default=200, type=int)
    parser.add_argument('--th', help='terminate threshold', default=100, type=int)

    args = parser.parse_args()

    formatted_pdf_folder = args.tar
    transcript_folder = args.qf
    result_folder = args.rs

    accepted_threshold = args.act
    pending_threshold = args.pdt

    align_level = args.lv
    match = args.mat
    mis_match = args.mma
    gap_pen = args.gp

    skip_query = args.sq
    cut_thres = args.cuts
    starting_search_space = args.sp
    cumulate = args.cml
    terminate_threshold = args.th

    # create a folder aligner with created path
    folalign = FolderAligner(formatted_pdf_folder, transcript_folder, result_folder, accepted_threshold, pending_threshold, match, mis_match, gap_pen, align_level, skip_query, cut_thres, starting_search_space, cumulate, terminate_threshold)

    # align the folder
    folalign.align_folder()