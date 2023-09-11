# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import os
import pandas as pd
import uncert as uc


# read mapping csv
df = pd.read_csv('process_videos_info.csv')

for index, row in df.iterrows():
    print(row['in'])
    print(row['out'])
    in_file = os.path.join(uc.common.get_configs('path_source'), row['in'])  # noqa: E501
    out_file = os.path.join(uc.common.get_configs('path_stimuli'), row['out'])  # noqa: E501
    # using https://superuser.com/a/859075/52383
    os.system("ffmpeg -r 30" +
              " -ss " + str(row['start']) +
              " -to " + str(row['end']) +
              " -i " + in_file +
              " -vcodec libx265" +
              " -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p" +  # noqa: E501
              " -c:a aac -ac 2 -b:a 128k" +
              " -movflags faststart" +
              " -vf scale=1280:720" +
              " -an " +
              " -crf 18 " +
              out_file)
