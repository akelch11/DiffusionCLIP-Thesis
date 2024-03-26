import subprocess
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--region", required=True, type=str)
    parser.add_argument("--big", type=int, default=0)
    args = parser.parse_args()
    classes = [
                # 'house', 
               'spices', 
               'religious_building', 
               'hand_soap',
                'dustbin',
                "medicine",
                ]
    regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]

    if args.big == 1:
        p_set = "BIG"
    else:
        p_set = "NORMAL"

    for class_name in classes:
        subprocess.run(
            [
                "sbatch",
                f"slurm/geode/interpolate/interpolate_geode_{args.region}_{class_name}_{p_set}.slurm"
            ]
        )
        print(f'submitting interpolate job for: geode {args.region} {class_name}')