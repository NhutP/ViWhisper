from num2words import num2words
from vietnam_number import n2w
import re
from pdfminer.high_level import extract_text
from num2words import num2words
from vietnam_number import n2w
import re
from tqdm import tqdm

import pathlib
import argparse

number_replacer = lambda x : num2words(int(x), lang='vi')

def romannum_integer(num):
    '''
    convert roman numeral to integer
    '''
    roman_numerals = {"I" : 1,
                    "V" : 5,
                    "X" : 10,
                    "L" : 50,
                    "C" : 100,
                    "D" : 500,
                    "M" : 1000
                    }

    int_value = 0
    num = num.strip()
    num = num.upper()
    for i in range(len(num)):
        if num[i] in roman_numerals:
            if i + 1 < len(num) and roman_numerals[num[i]] < roman_numerals[num[i + 1]]:
                int_value -= roman_numerals[num[i]]
            else:
                int_value += roman_numerals[num[i]]
        else:
            print("Invalid input.")
            quit()

    return str(int_value)


def read_number(x): 
    '''
    convert number to words
    '''           
    x = x.replace('.', '')
    x = x.replace(' ', '')

    if ',' in x:
        a , b = x.split(',')
        return number_replacer(a) + ' phẩy ' + number_replacer(b)
    else:
        return number_replacer(x)

def read_day_month_year(date):
    d,m,y = date.split('/')
    return f'{number_replacer(d)} tháng {number_replacer(m)} năm {number_replacer(y)}'

def read_month_year(date):
    m , y = date.split('/')
    return f'{number_replacer(m)} năm {number_replacer(y)}'

def read_day_month(date):
    d, m = date.split('/')
    return f'{number_replacer(d)} tháng {number_replacer(m)}'

class Replacer:
    def __init__(self, chars_to_eliminate):
        self.chars_to_eliminate = chars_to_eliminate

        # regex for number
        self.number_regex = r"((?:\d+(?:(?:\.| )))*(?:\d+))(,(?:\d+(?:(?:\.|\s)))*(?:\d+))?"
        # regex for roman numeral
        self.roman_numeral_regex = r"\s[MDCLVXI]+\s"
        # regex for dates
        self.dd_mm_yy_regex = r"\d+\/\d+\/\d+"
        self.mm_yy_regex = r"(((?<!\d)(0?[1-9]))|(1[0-2]))\/(\d+)"
        self.dd_mm_regex = r"(\d+\/\d+)"
        # regex for URL
        self.url_regex = r"((?:(?:https?|ftp):\/\/)[\w\/\-?=%.]+\.[\w\/\-&?=%.]+)|(www\.[\w\/\-?=%.]+\.[\w\/\-&?=%.]+)"

    
        self.roman_integer = romannum_integer
        # replace number with words
        self.read_number = read_number
        # replace dates with words
        self.read_day_month_year = read_day_month_year
        self.read_month_year = read_month_year
        self.read_day_month = read_day_month


    def process_num(self, text):
        '''
        replace numbers with words
        '''
        text = re.sub(self.number_regex, lambda x : self.read_number(x.group()),text)
        return text
    

    def process_roman_num(self, text):
        '''
        replace roman numerals with words
        '''
        text = re.sub(self.roman_numeral_regex, lambda x : ' ' + self.read_number(self.roman_integer(x.group().strip())) + ' ', text)
        return text
        

    def process_dd_mm_yy(self, text):
        text = re.sub(self.dd_mm_yy_regex, lambda x : self.read_day_month_year(x.group()), text)
        return text
    

    def process_dd_mm(self, text):
        text = re.sub(self.dd_mm_regex, lambda x : self.read_day_month(x.group()), text)
        return text
    

    def process_mm_yy(self, text):
        text = re.sub(self.mm_yy_regex, lambda x : self.read_month_year(x.group()), text)
        return text
    

    def process_url(self, text):
        '''
        eliminate URL
        '''
        text = re.sub(self.url_regex, ' ', text)
        return text


    def eliminate_character(self, text: str):
        '''
        eliminate required characters
        '''
        for char in self.chars_to_eliminate:
            text = text.replace(char, ' ')
        return text
    

    def replace_symbols(self, text):
        text = text.replace('%', ' phần trăm ')
        text = text.replace('@', ' a còng ')
        text = text.replace('#', ' thăng ')
        text = text.replace('+', ' cộng ')
        text = text.replace('$', ' đô la ')
        text = text.replace('>', ' lớn hơn ')
        text = text.replace('<', ' bé hơn ')
        text = text.replace('w', ' vê kép ')
        return text    
    

    def delete_space(self, text: str):
        text = re.sub(r'\s+', ' ', text)
        return text.lower()
    

    def vn_words(self, text):
        '''
        only allow vietnamese characters
        '''
        text = re.sub(r'[^a-z0-9A-Z_àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]', ' ', text)
        return text
    

    def replace_all(self, text):
        text = self.process_url(text)
        text = self.process_dd_mm_yy(text)
        text = self.process_mm_yy(text)
        text = self.process_dd_mm(text)
        text = text.replace('\n', ';')
        text = self.process_num(text)
        text = self.eliminate_character(text)
        text = self.replace_symbols(text)
        text = self.process_roman_num(text)
        text = self.vn_words(text)
        text = self.delete_space(text)
        return text
    

class PDFLoader(Replacer):
    def __init__(self, path_to_eliminate_chars):
        with open(path_to_eliminate_chars, 'r', encoding='utf8') as f:
            chars_to_eliminate = f.read()
        super().__init__(chars_to_eliminate)
        

    def load(self, path):
        return extract_text(path)
    

    def load_and_process_pdf(self, pdf_file_path, result_file_path):
        pdf_text = self.load(pdf_file_path)
        pdf_text = self.replace_all(pdf_text)

        if not pathlib.Path(result_file_path).parent.exists():
            pathlib.Path(result_file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(result_file_path, 'w', encoding='utf8') as pdf_writer:
            pdf_writer.write(pdf_text)


    def load_and_process_text(self, text_file_path, result_file_path):
        with open(text_file_path, 'r', encoding='utf8') as f:
            raw_text = f.read()

        raw_text = self.replace_all(raw_text)

        if not pathlib.Path(result_file_path).parent.exists():
            pathlib.Path(result_file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(result_file_path, 'w', encoding='utf8') as pdf_writer:
            pdf_writer.write(raw_text)


    def pdf_process_folder(self, path_to_book_folder: pathlib.Path, path_to_result_folder: pathlib.Path, crp=None):
        assert path_to_book_folder.exists() == True

        print("Formatting pdf files")
        corrupted_file = []

        for pdf_file in path_to_book_folder.rglob('*.pdf'):
            print(f'Formating {pdf_file}')
            pdf_file_name = pdf_file.stem
            path_to_formatted_file = pathlib.Path(str(pdf_file.parent).replace(str(path_to_book_folder), str(path_to_result_folder))) / (pdf_file_name + '.txt')

            if not path_to_formatted_file.exists():                    
                try:
                    self.load_and_process_pdf(pdf_file, path_to_formatted_file)
                    print(f'Completed {pdf_file_name}\n-------------------------------------')
                except Exception as E:
                    print(E)
                    print(f'Can NOT read {pdf_file_name} \n----------------------------------------')
                    corrupted_file.append(str(path_to_formatted_file))
                    continue
            else:
                print(f'Already completed {pdf_file_name}\n--------------------------------')

        if crp:
            path_to_corrupted_pdf = pathlib.Path(crp)
            with open(path_to_corrupted_pdf, 'a') as f:
                for file in corrupted_file:
                    f.write(file  + '\n')
            print(f"There are {len(corrupted_file)} corrupted pdf files, stored in {path_to_corrupted_pdf}")
        else:
            print(f"There are {len(corrupted_file)} corrupted pdf files")


    def text_process_folder(self, path_to_book_folder: pathlib.Path, path_to_result_folder: pathlib.Path):
        assert path_to_book_folder.exists() == True

        print("Formatting txt file")

        for txt_file in path_to_book_folder.rglob('*.txt'):
            print(f'Formating {txt_file}')
            txt_file_name = txt_file.stem
            path_to_formatted_file = pathlib.Path(str(txt_file).replace(str(path_to_book_folder), str(path_to_result_folder)))

            if not path_to_formatted_file.exists():                    
                self.load_and_process_text(txt_file, path_to_formatted_file)
                print(f'Completed {txt_file_name}\n-------------------------------------')
            else:
                print(f'Already completed {txt_file_name}\n--------------------------------')


    def process_all_folder(self, path_to_book_folder: pathlib.Path, path_to_result_folder: pathlib.Path, crp=None):
        self.pdf_process_folder(path_to_book_folder, path_to_result_folder, crp)
        self.text_process_folder(path_to_book_folder, path_to_result_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ec', help='eliminate character', type=str)
    parser.add_argument('rb', help='path to raw book pdf', type=str)
    parser.add_argument('re', help='path to result folder', type=str)
    parser.add_argument('--crp', help= 'text file for corrupted pdf', type=str)
    args = parser.parse_args()

    pdf_loader = PDFLoader(pathlib.Path(args.ec))
    path_to_book_folder = pathlib.Path(args.rb)
    path_to_result_folder = pathlib.Path(args.re)

    path_to_corrupted_pdf = pathlib.Path(args.crp) if args.crp is not None else None

    pdf_loader.process_all_folder(path_to_book_folder, path_to_result_folder, path_to_corrupted_pdf)