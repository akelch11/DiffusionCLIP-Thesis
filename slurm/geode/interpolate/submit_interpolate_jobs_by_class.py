import subprocess
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--class_name", required=True, type=str)
    parser.add_argument("--big", type=int, default=0)
    args = parser.parse_args()
    classes = [
                'house', 
               'spices', 
               'religious_building', 
               'hand_soap',
                'dustbin',
                "medicine"
                ]
    regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]

    if args.big == 1:
        p_set = "BIG"
    else:
        p_set = "NORMAL"

    for region in regions:
        subprocess.run(
            [
                "sbatch",
                f"slurm/geode/interpolate/interpolate_geode_{region}_{args.class_name}_{p_set}.slurm"
            ]
        )
        print(f'submitting interpolate job for: geode {region} {args.class_name}')