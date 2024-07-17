def gen_vocab(src_path, vocab_path):
    lexicon = {}
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            idx = parts[0]
            phones = parts[1:]

            for ps in phones:
                for p in ps:
                    if p not in lexicon:
                        lexicon[p] = 1
                    else:
                        lexicon[p] += 1

    vocab = sorted(lexicon.items(), key=lambda x: x[1], reverse=True)

    index = 3
    with open(vocab_path, 'w') as w:
        w.write('<PAD> 0\n')
        w.write('<S/E> 1\n')
        w.write('<UNK> 2\n')
        for (l, n) in vocab:
            w.write(l+' '+str(index)+'\n')
            index += 1