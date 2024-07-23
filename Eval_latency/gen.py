from pypinyin import lazy_pinyin, Style
import shutil

def text2pinyin(path, goal_path, withidx = True):
    lines = []

    idx2pinyin = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            idx = parts[0]
            phones = parts[1:]

            str1 = ""
            for ps in phones:
                str1 += ps

            strs = lazy_pinyin(str1,style = Style.TONE3)
            str_pinyin=""
            for pinyin in strs:
                str_pinyin += pinyin + " " 
            
            idx2pinyin[idx] = str_pinyin

            if withidx:
                str_pinyin = idx + " " + str_pinyin
    
            lines.append(str_pinyin)


    with open(goal_path, 'w') as w:
        for sub_str in lines:
            w.write(sub_str + "\n")
    
    return idx2pinyin

def gen_wav_text(list1):
    # wav2path = {}
    # with open("egs/aishell/data/test/wav.scp", 'r', encoding='utf-8') as fid:
    #     for line in fid:
    #         idx, path = line.strip().split()
    #         path = "./egs/aishell/"+path
    #         wav2path[idx] = path
    
    # text2tg = {}
    # with open("egs/aishell/data/test/text", 'r', encoding='utf-8') as t:
    #     for line in t:
    #         parts = line.strip().split()
    #         utt_id = parts[0]
    #         label =  parts[1:]
    #         text2tg[utt_id] = label

    # for name in list1:
    #     shutil.copyfile(wav2path[name], "./1/"+name+".wav")
    #     phones = text2tg[name]
    #     str1 = ""
    #     for ps in phones:
    #         str1 += ps

    #     strs = lazy_pinyin(str1,style = Style.TONE3)
    #     str_pinyin=""
    #     for pinyin in strs:
    #         str_pinyin += pinyin + " " 

    #     with open("./1/"+name+".lab", 'w', encoding='utf-8') as w:
    #         w.write(str_pinyin)

    # with open("./Eval_latency/test/text", 'w', encoding='utf-8') as w:
    #     for name in list1:
    #         phones = text2tg[name]
    #         str1 = ""
    #         for ps in phones:
    #             str1 += ps+" "
    #         w.write(name+"     "+str1 + "\n")

    wav2path = {}
    with open("egs/aishell/data/test/wav.scp", 'r', encoding='utf-8') as fid:
        for line in fid:
            idx, path = line.strip().split()
            wav2path[idx] = path

    with open("./Eval_latency/test/wav.scp", 'w', encoding='utf-8') as w:
        for name in list1:
            w.write(name+" "+wav2path[name] + "\n")

