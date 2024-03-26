

classes = ['house', 'spices', 'religious_building', 'hand_soap', 'dustbin', "medicine"]
regions = ["Africa", "Americas", "EastAsia", "Europe", "SouthEastAsia", "WestAsia"]

region_format = {k:k for k in regions}
region_format['EastAsia']= "East Asia"
region_format["SouthEastAsia"]="Southeast Asia"
region_format["WestAsia"] = "West Asia"




with open('new_clip_text_pairs.txt', "w") as text_file:
    for class_name in classes:
        for region in regions:

            fmt_class = class_name
            fmt_class.replace("_", " ")
        

            fmt_region = region_format[region]

            pair_name = f"\"{class_name}_in_{region}\": ([\"{fmt_class}\"], [\"A {fmt_class} in {fmt_region}\"]),\n"
            text_file.write(pair_name)
                