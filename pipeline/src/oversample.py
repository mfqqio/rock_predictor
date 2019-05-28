import pandas as pd
import random
import argparse, sys

if len(sys.argv)<3: # then running in dev mode (can remove in production)
    input_train = "data/train.csv"
    output_train = "data/train-oversample-100.csv"
    id_col = "hole_id"
    class_col = "litho_rock_class"
    extra_samples = {"QZ":100}

else: # parse input parameters from terminal or makefile
    parser = argparse.ArgumentParser()
    parser.add_argument("input_train")
    parser.add_argument("output_train")
    parser.add_argument("id_col")
    parser.add_argument("class_col")
    parser.add_argument("extra_samples")
    args = parser.parse_args()

    input_train = args.input_train
    output_test = args.output_train
    id_col = args.id_col
    class_col = args.class_col
    extra_samples = args.extra_samples

random.seed(123)

df = pd.read_csv(input_train)
print("input shape:",df.shape)
df["oversample"] = False

df_samples = pd.DataFrame()

for key, val in extra_samples.items():
    ids = df[id_col][df[class_col]==key].unique().tolist()
    extra_sample_ids = random.choices(ids, k=val) ## with replacement
    for id in extra_sample_ids:
        df_samples = df_samples.append(df[df[id_col]==id])
df_samples["oversample"] = True
print("extra samples shape:",df_samples.shape)
df = df.append(df_samples, ignore_index=True, sort=False)
print("output shape:",df.shape)

print("Saving output file...")
df.to_csv(output_train, index=False)
print("Oversampling complete!")
