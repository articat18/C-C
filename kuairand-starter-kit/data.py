"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = [
    'user_id',
    'video_id',
    'author_id',
    'tab',
    'dur_bucket',
    'user_activity',
    'video_popularity',  
]

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out
def add_history_features(splits):
    """
    Build additional categorical features using TRAIN only.

    Features:
      user_activity
      video_popularity
      author_popularity
      user_longview_rate
      video_longview_rate

    All statistics are calculated from TRAIN only,
    preventing validation/test label leakage.
    """

    train = splits['train']
    duration_edges = _bucket_edges(
    [x[5] for x in train],
    n=10)
    

    # --------------------------------------------------
    # 1. Count impressions and positive labels
    # --------------------------------------------------

    user_imp = collections.Counter()
    user_pos = collections.Counter()

    video_imp = collections.Counter()
    video_pos = collections.Counter()

    author_imp = collections.Counter()
    author_pos = collections.Counter()

    for x in train:

        user = x[1]
        video = x[2]
        author = x[3]
        label = x[6]

        user_imp[user] += 1
        user_pos[user] += label

        video_imp[video] += 1
        video_pos[video] += label

        author_imp[author] += 1
        author_pos[author] += label

    # --------------------------------------------------
    # 2. Create quantile bucket boundaries
    # --------------------------------------------------

    user_imp_edges = _bucket_edges(
        list(user_imp.values()),
        n=10
    )

    video_imp_edges = _bucket_edges(
        list(video_imp.values()),
        n=10
    )

    author_imp_edges = _bucket_edges(
        list(author_imp.values()),
        n=10
    )


    # --------------------------------------------------
    # 3. Attach features to every split
    # --------------------------------------------------

    out = {}

    for name, rows in splits.items():

        new_rows = []

        for x in rows:

            date = x[0]
            user = x[1]
            video = x[2]
            author = x[3]
            tab = x[4]
            duration = x[5]
            label = x[6]

            # ------------------------------------------
            # Training statistics
            # ------------------------------------------

            ui = user_imp.get(user, 0)
            vi = video_imp.get(video, 0)
            ai = author_imp.get(author, 0)

            up = user_pos.get(user, 0)
            vp = video_pos.get(video, 0)

            user_rate = (
                up / ui
                if ui > 0
                else 0.0
            )

            video_rate = (
                vp / vi
                if vi > 0
                else 0.0
            )

            # ------------------------------------------
            # Convert continuous statistics to buckets
            # ------------------------------------------

            user_activity = str(
                int(
                    np.searchsorted(
                        user_imp_edges,
                        ui
                    )
                )
            )

            video_popularity = str(
                int(
                    np.searchsorted(
                        video_imp_edges,
                        vi
                    )
                )
            )

            author_popularity = str(
                int(
                    np.searchsorted(
                        author_imp_edges,
                        ai
                    )
                )
            )


            # ------------------------------------------
            # Duration bucket
            # ------------------------------------------

            dur_bucket = str(
                int(
                    np.searchsorted(
                    duration_edges,
                    duration
                    )
                )
            )

            # ------------------------------------------
            # Explicit interaction features
            # ------------------------------------------

            user_tab = user + '_' + tab

            user_author = user + '_' + author

            video_tab = video + '_' + tab

            user_dur = user + '_' + dur_bucket

            # ------------------------------------------
            # New row
            # ------------------------------------------

            new_rows.append(
                (
                    date,
                    user,
                    video,
                    author,
                    tab,
                    duration,
                    label,

                    user_activity,
                    video_popularity,
                    author_popularity,


                    user_tab,
                    user_author,
                    video_tab,
                    user_dur,
                )
            )

        out[name] = new_rows

    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """
    Convert categorical features into continuous integer IDs.

    Returns:
        enc[name] = (X, y, users)

    X shape:
        (N, len(FIELDS))
    """

    tr = splits['train']

    # Duration boundaries from TRAIN only
    duration_edges = _bucket_edges(
        [x[5] for x in tr],
        n=10
    )

    def raw(x):

        user = x[1]
        video = x[2]
        author = x[3]
        tab = x[4]
        duration = x[5]

        dur_bucket = str(
            int(
                np.searchsorted(
                    duration_edges,
                    duration
                )
            )
        )
        

        return [
            user,
            video,
            author,
            tab,
            dur_bucket,

            x[7],   # user_activity
            x[8],   # video_popularity
           
        ]

    # --------------------------------------------------
    # Build vocabularies using TRAIN only
    # --------------------------------------------------

    vocabs = [
        dict()
        for _ in FIELDS
    ]

    for x in tr:

        values = raw(x)

        for i, v in enumerate(values):

            if v not in vocabs[i]:

                vocabs[i][v] = len(
                    vocabs[i]
                )

    # UNK ID for every feature field
    unk = [
        len(v)
        for v in vocabs
    ]

    field_dims = [
        len(v) + 1
        for v in vocabs
    ]

    # Make every field occupy a different ID range
    offsets = np.cumsum(
        [0] + field_dims[:-1]
    ).astype(np.int32)

    # --------------------------------------------------
    # Encode each split
    # --------------------------------------------------

    enc = {}

    for name, rows in splits.items():

        X = np.empty(
            (
                len(rows),
                len(FIELDS)
            ),
            dtype=np.int32
        )

        y = np.empty(
            len(rows),
            dtype=np.float32
        )

        users = []

        for n, x in enumerate(rows):

            values = raw(x)

            for i, v in enumerate(values):

                X[n, i] = (
                    vocabs[i].get(
                        v,
                        unk[i]
                    )
                    + offsets[i]
                )

            y[n] = x[6]

            users.append(x[1])

        enc[name] = (
            X,
            y,
            users
        )

    return enc, int(sum(field_dims))
