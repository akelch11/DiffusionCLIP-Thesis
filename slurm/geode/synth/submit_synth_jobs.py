import subprocess
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--region", required=True, type=str)
    parser.add_argument("--big", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0)
    args = parser.parse_args()
    classes = [
                # 'house', 
            #    'spices', 
            #    'religious_building', 
               'hand_soap',
                'dustbin',
                "medicine",
                # "car", 
                # "plate_of_food",
                # "storefront"
                ]
    regions = [
                "Africa",
                "Americas",
                  "EastAsia",
                    "Europe", "SouthEastAsia", "WestAsia"]

    if args.big == 1:
        p_set = "BIG"
    else:
        p_set = "NORMAL"


    eta_str = str(args.eta).replace(".","d")



    for class_name in classes:
        for eta_str in ["0d0","0d2"]:
            subprocess.run(
                [
                    "sbatch",
                    f"slurm/geode/synth/synth_geode_{args.region}_{class_name}_{p_set}_b{eta_str}.slurm"
                ]
            )
            print(f'submitting synth job for: geode {args.region} {class_name} eta: {eta_str.replace('d', '.')}')