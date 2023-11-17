import csv
from time import time
from konlpy import tag
from konlpy.corpus import kolaw
from konlpy.utils import  pprint


def custom_csv_write(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


def tagging(tagger, text):
    r = []
    try:
        r = getattr(tag, tagger)().pos(text)
    except Exception as e:
        print("Uhoh,", e)
    return r

def measure_time(taggers, mult=6):
    doc = kolaw.open('constitution.txt').read() * mult
    data = [['n'] + taggers]
    for i in range(mult):
        doclen = 10 ** i
        times = [time()]
        diffs = [doclen]
        for tagger in taggers:
            r = tagging(tagger, doc[:doclen])
            times.append(time())
            diffs.append(times[-1] - times[-2])
            print('%s\t%s\t%s' % (tagger[:5], doclen, diffs[-1]))
            pprint(r[:5])
        data.append(diffs)
        print()
    return data

def measure_accuracy(taggers, text):
    print('\n%s' % text)
    result = []
    for tagger in taggers:
        print(tagger, end=' ')
        r = tagging(tagger, text)
        pprint(r)
        result.append([tagger] + [' / '.join(s) for s in r])
    return result

def plot(result):
    from matplotlib import pylab as pl
    import scipy as sp

    if not result:
        result = sp.loadtxt('morph.csv', delimiter=',', skiprows=1).T

    x, y = result[0], result[1:]
    for i in y:
        pl.plot(x, i)

    pl.xlabel('Number of characters')
    pl.ylabel('Time (sec)')
    pl.xscale('log')
    pl.grid(True)
    pl.savefig("images/time.png")
    pl.show()

if __name__ == '__main__':
    PLOT = False
    MULT = 6

    examples = ['아버지가방에들어가신다',  # 띄어쓰기
                '나는 밥을 먹는다', '하늘을 나는 자동차', # 중의성 해소
                '아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다 6+ 128기가실버ㅋ'] # 속어

    taggers = [t for t in dir(tag) if t[0].isupper()]

    # Time
    data = measure_time(taggers, MULT)
    # custom_csv_write(data, 'morph.csv')

    # Accuracy
    for i, example in enumerate(examples):
        result = measure_accuracy(taggers, example)
        result = list(zip(*result))

    # Plot
    if PLOT:
        plot(result)
