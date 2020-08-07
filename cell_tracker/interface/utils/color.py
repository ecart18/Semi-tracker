# -*- coding: UTF-8 -*-

from __future__ import absolute_import

"""
    背景色：深灰
    group1：Snow Seashell AntiqueWhite
    group2：Bisque PeachPuff NavajoWhite
    group3：LemonChiffon Cornsilk Ivory
    group4：Honeydew LavenderBlush MistyRose
    group5：Azure SlateBlue RoyalBlue
    group6：Blue DodgerBlue SteelBlue
    group7：DeepSkyBlue SkyBlue LightSkyBlue
    group8：SlateGray lightSteelBlue LightBlue
    group9：LightCyan PaleTurquoise CadetBlue
    group10：Turquoise Cyan DarkSlateGray
    group11：Aquamarine DarkSeaGreen SeaGreen
    group12：PaleGreen, SpringGreen, Green
    group13：Chartreuse OliveDrab DarkOliveGreen
    group14：Khaki LightGoldenrod LightYellow
    group15：Yellow Gold Goldenrod
    group16：DarkGoldenrod RosyBrown IndianRed
    group17：Sienna Burlywood Wheat
    group18：Tan Chocolate Firebrick
    group19：Brown Salmon LightSalmon
    group20：Orange DarkOrange Coral
    group21：Tomato OrangeRed Red
    group22：DeepPink HotPink Pink
    group23：LightPink PaleVioletRed Maroon
    group24：VioletRed Magenta Orchid
    group25：Plum MediumOrchid DarkOrchid
    group26：Purple MediumPurPle Thistle
"""

color_group1 = [[255, 250, 250], [238, 233, 233], [205, 201, 201], [139, 137, 137],
                [255, 245, 238], [238, 229, 222], [205, 197, 191], [139, 134, 130],
                [255, 239, 219], [238, 223, 204], [205, 192, 176], [139, 131, 210]
                ]
color_group2 = [[255, 228, 196], [238, 213, 183], [205, 183, 158], [139, 125, 107],
                [255, 218, 185], [238, 203, 173], [205, 175, 149], [139, 119, 101],
                [255, 222, 173], [238, 207, 161], [205, 179, 139], [139, 121,  94]
                ]
color_group3 = [[255, 250, 205], [238, 233, 191], [205, 201, 165], [139, 137, 112],
                [255, 248, 220], [238, 232, 205], [205, 200, 177], [139, 136, 120],
                [255, 255, 240], [238, 238, 224], [205, 205, 193], [139, 139, 131],
                ]
color_group4 = [[240, 255, 240], [224, 238, 224], [193, 205, 193], [131, 139, 131],
                [255, 240, 245], [238, 224, 229], [205, 193, 197], [139, 131, 134],
                [255, 228, 225], [238, 213, 210], [205, 183, 181], [139, 125, 123]
                ]
color_group5 = [[240, 255, 255], [224, 238, 238], [193, 205, 205], [131, 139, 139],
                [131, 111, 255], [122, 103, 238], [105,  89, 205], [ 71,  60, 139],
                [ 72, 118, 255], [ 67, 110, 238], [ 58,  95, 205], [ 39,  64, 139]
                ]
color_group6 = [[  0,   0, 255], [  0,   0, 238], [  0,   0, 205], [  0,   0, 139],
                [ 30, 144, 255], [ 28, 134, 238], [ 24, 116, 205], [ 16,  78, 139],
                [ 99, 184, 255], [ 92, 172, 238], [ 79, 148, 205], [ 54, 100, 139]
                ]
color_group7 = [[  0, 191, 255], [  0, 178, 238], [  0, 154, 205], [  0, 104, 139],
                [135, 206, 255], [126, 192, 238], [108, 166, 205], [ 74, 112, 139],
                [176, 226, 255], [164, 211, 238], [141, 182, 205], [ 96, 123, 139]
                ]
color_group8 = [[198, 226, 255], [185, 211, 238], [159, 182, 205], [108, 123, 139],
                [202, 225, 255], [188, 210, 238], [162, 181, 205], [110, 123, 139],
                [191, 239, 255], [178, 223, 238], [154, 192, 205], [104, 131, 139]
                ]
color_group9 = [[224, 255, 255], [209, 238, 238], [180, 205, 205], [122, 139, 139],
                [187, 255, 255], [174, 238, 238], [150, 205, 205], [102, 139, 139],
                [152, 245, 155], [142, 229, 238], [122, 197, 205], [ 83, 134, 139]
                ]
color_group10 = [[  0, 245, 255], [  0, 229, 238], [  0, 197, 205], [  0, 134, 139],
                [  0, 255, 255], [  0, 238, 238], [  0, 205, 205], [  0, 139, 139],
                [151, 255, 255], [141, 238, 238], [121, 205, 205], [ 82, 139, 139]
                ]
color_group11 = [[127, 255, 212], [118, 238, 198], [102, 205, 170], [ 69, 139, 116],
                [193, 255, 193], [180, 238, 180], [155, 205, 155], [105, 139, 105],
                [ 84, 255, 159], [ 78, 238, 148], [ 67, 205, 128], [ 46, 139,  87]
                ]
color_group12 = [[154, 255, 154], [144, 238, 144], [124, 205, 124], [ 84, 139,  84],
                [  0, 255, 127], [  0, 238, 118], [  0, 205, 102], [  0, 139,  69],
                [  0, 255,   0], [  0, 238,   0], [  0, 205,   0], [  0, 139,   0]
                ]
color_group13 = [[127, 255,   0], [118, 238,   0], [102, 205,   0], [ 69, 139,   0],
                [192, 255,  62], [179, 238,  58], [154, 205,  50], [105, 139,  34],
                [202, 255, 112], [188, 238, 104], [162, 205,  90], [110, 139,  61]
                ]
color_group14 = [[255, 246, 143], [238, 230, 133], [205, 198, 115], [139, 134,  78],
                [255, 236, 139], [238, 220, 130], [205, 190, 112], [139, 129,  76],
                [255, 255, 224], [238, 238, 209], [205, 205, 180], [139, 139, 122]
                ]
color_group15 = [[255, 255,   0], [238, 238,   0], [205, 205,   0], [139, 139,   0],
                [255, 215,   0], [238, 201,   0], [205, 173,   0], [139, 117,   0],
                [255, 193,  37], [238, 180,  34], [205, 155,  29], [139, 105,  20]
                ]
color_group16 = [[255, 185,  15], [238, 173,  14], [205, 149,  12], [139, 101,   8],
                [255, 193, 193], [238, 180, 180], [205, 155, 155], [139, 105, 105],
                [255, 106, 106], [238,  99,  99], [205,  85,  85], [139,  58,  58]
                ]
color_group17 = [[255, 130,  71], [238, 121,  66], [205, 104,  57], [139,  71,  38],
                [255, 211, 155], [138, 197, 145], [205, 170, 125], [139, 115,  85],
                [255, 231, 186], [238, 216, 174], [205, 186, 150], [139, 126, 102]
                ]
color_group18 = [[255, 165,  79], [238, 154,  73], [205, 133,  63], [139,  90,  43],
                [255, 127,  36], [238, 118,  33], [205, 102,  29], [139,  69,  19],
                [255,  48,  48], [238,  44,  44], [205,  38,  38], [139,  26,  26],
                ]
color_group19 = [[255,  64,  64], [238,  59,  59], [205,  51,  51], [139,  35,  35],
                [255, 140, 105], [238, 130,  98], [205, 112,  84], [139,  76,  57],
                [255, 160, 122], [238, 149, 114], [205, 129,  98], [139,  87,  66]
                ]
color_group20 = [[255, 165,   0], [238, 154,   0], [205, 133,   0], [139,  90,   0],
                [255, 127,   0], [238, 118,   0], [205, 102,   0], [139,  69,   0],
                [255, 114,  86], [238,  106, 86], [205,  91,  69], [139,  54,  38]
                ]
color_group21 = [[255,  90,  71], [238,  92,  66], [205,  79,  57], [139,  54,  38],
                [255,  69,   0], [238,  64,   0], [205,  55,   0], [139,  37,   0],
                [255,   0,   0], [238,   0,   0], [205,   0,   0], [139,   0,   0]
                ]
color_group22 = [[255,  20, 147], [238,  18, 137], [205,  16, 118], [139,  10,  80],
                [255, 110, 180], [238, 106, 167], [205,  96, 144], [139,  58,  98],
                [255, 181, 197], [238, 169, 184], [205, 145, 158], [139,  99, 108]
                ]
color_group23 = [[255, 174, 185], [238, 162, 173], [205, 140, 149], [139, 95, 101],
                [255, 130, 171], [238, 121, 159], [205, 104, 137], [139,  71, 93],
                [255,  52, 179], [238,  48, 167], [205,  41, 144], [139,  28, 98]
                ]
color_group24 = [[255,  62, 150], [238,  58, 140], [205,  50, 120], [139,  34,  82],
                [255,   0, 255], [238,   0, 238], [205,   0, 205], [139,   0, 139],
                [255, 131, 205], [238, 122, 233], [205, 105, 201], [139,  71, 137]
                ]
color_group25 = [[255, 187, 255], [238, 174, 238], [205, 150, 205], [139, 102, 139],
                [224, 102, 255], [209,  95, 238], [180,  82, 205], [122,  55, 139],
                [191,  62, 255], [178,  58, 238], [154,  50, 205], [104,  34, 139]
                ]
color_group26 = [[155,  48, 255], [145,  44, 238], [125,  38, 205], [ 85,  26, 139],
                [171, 130, 255], [159, 121, 238], [137, 104, 205], [139, 123, 139],
                [255, 225, 255], [238, 210, 238], [205, 181, 205], [139, 123, 139]
                ]

color_groups = [color_group1, color_group2, color_group3, color_group4,
                color_group5, color_group6, color_group7, color_group8, 
                color_group9, color_group10, color_group11, color_group12, 
                color_group13, color_group14, color_group15, color_group16, 
                color_group17, color_group18, color_group19, color_group20, 
                color_group21, color_group22, color_group23, color_group24,
                color_group25, color_group26]

annotation_colors = [color_group1[0], color_group2[0], color_group3[0], color_group4[0],
                     color_group5[0], color_group6[0], color_group7[0], color_group8[0],
                     color_group9[0], color_group10[0], color_group11[0], color_group12[0],
                     color_group13[0], color_group14[0], color_group15[0], color_group16[0],
                     color_group17[0], color_group18[0], color_group19[0], color_group20[0],
                     color_group21[0], color_group22[0], color_group23[0], color_group24[0],
                     color_group25[0], color_group26[0]]
