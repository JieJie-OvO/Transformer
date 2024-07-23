from pypinyin import lazy_pinyin, Style

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