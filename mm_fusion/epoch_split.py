"""Where does one epoch go? Parse tqdm completion lines for real timings."""
import re
import os

LOGS = [
    ('2+2  (ours, 5 cat)', 'logs/mm_sep22_train.log'),
    ('3+3  (ours, 7 cat)', 'logs/mm_sep33_train.log'),
    ('4+4  (ours, 9 cat)', 'logs/mm_sep44_train.log'),
    ('text only (legacy, 5 cat)', 'logs/beauty_e1_qwen3_vl_8b_train.launcher.log'),
    ('legacy 2+2 (5 cat)', 'logs/beauty_e4_freqmax_pack2.log'),
]

TRAIN = re.compile(r'Training - \[Epoch (\d+)\]: 100%\|[^|]*\| (\d+)/(\d+) \[([0-9:]+)<')
VAL = re.compile(r'Eval - val: 100%\|[^|]*\| (\d+)/(\d+) \[([0-9:]+)<')


def secs(t):
    parts = [int(x) for x in t.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


os.chdir('/home/lzr/science/actionpiece_exp')
print('%-26s %-8s %-10s %-8s %-10s %-9s %s'
      % ('arm', 'tr_batch', 'tr_time', 'va_batch', 'va_time', 'va_share', 'epoch'))
print('-' * 88)
for label, path in LOGS:
    if not os.path.exists(path):
        print('%-26s (no log)' % label)
        continue
    text = open(path, errors='ignore').read().replace('\r', '\n')
    tr = TRAIN.findall(text)
    va = VAL.findall(text)
    if not tr or not va:
        print('%-26s incomplete (train bars %d, val bars %d)'
              % (label, len(tr), len(va)))
        continue
    # last completed epoch's training bar, and a representative val bar
    ep, tb, tn, tt = tr[-1]
    tb, tn = int(tb), int(tn)
    tsec = secs(tt)
    vb, vn, vt = va[-1]
    vb, vn = int(vb), int(vn)
    vsec = secs(vt)
    total = tsec + vsec
    print('%-26s %-8d %-10s %-8d %-10s %-9s %s'
          % (label, tn, '%d:%02d' % (tsec // 60, tsec % 60),
             vn, '%d:%02d' % (vsec // 60, vsec % 60),
             '%.0f%%' % (100 * vsec / total),
             '%d:%02d' % (total // 60, total % 60)))

print()
print('tr_batch / va_batch = number of batches per epoch')
print('va_share            = fraction of the epoch spent validating')
