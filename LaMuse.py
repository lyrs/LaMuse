#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007
#

from glob import glob
from tkinter import Image

import numpy
import cv2
import errno
import argparse
import json

#LuisV
from tqdm import tqdm
import PIL.Image
from numpy import array

import pkg_resources

import PySimpleGUI as sg

#from .tools.color_palette import get_color_names
from .tools.generate_segmented_pictures import generate_images
from .tools.create_original_case_study import create_case_study
from .tools.fast_style_transfer import apply_style_transfer
from .tools.watermarking import add_watermark

from .Musesetup import *
import pandas as pd

sg.theme('DarkAmber')

layout = [[sg.Text("Dossier d'images substituts"), sg.Input(), sg.FolderBrowse(initial_folder=".")],
          # [gui.Text('Veuillez confirmer le traitement d\'images actuelles (ne pas fermer cette fenëtre durant le
          # traitement) : ')],
          [sg.Button("Génération de substituts")],
          [sg.Text('Dossier d\'oeuvres'), sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Dossier d\'images de fond : '), sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Button("Créer un cas d'étude")]
          ]


def generate_full_case_study(painting_folder: str, substitute_folder: str,
                             background_folder: str, interpretation_folder: str):
    """
    :param painting_folder:
    :param substitute_folder:
    :param background_folder:
    :param interpretation_folder:
    :return:
    """
    ##
    # The following function will go over all images in 'painting_folder' and use
    # the Mask_RCNN neural network to find identifiable objects.
    # It will then substitute these objects with similar ones stored in the 'substitute_folder'
    # folder, and replace the background with a random image chosen from 'background_folder'
    # The results are stored in 'interpretation_folder'
    ##
    if args.verbose:
        print("   Calling create_case_study")

    trace_log = create_case_study(painting_folder, substitute_folder, background_folder, interpretation_folder, 1,
                                  args.bw)

    if args.verbose:
        print("   Done calling create_case_study")

    ##
    # Go over all images in 'default_painting_folder' and the corresponding images in
    # 'default_interpretation_folder' and apply a style transfer.
    ##
    
    #painting_file_list = [y for x in [glob(f'{painting_folder}/*.{ext}', recursive=True) for ext in image_extensions]
    #                      for y in x]
    
    # LuisV
    # Get the list of all files in directory tree at given path
    painting_file_list = list()
    for (dirpath, dirnames, filenames) in os.walk(painting_folder):
        painting_file_list += [os.path.join(dirpath, file) for file in filenames]
    print(f">>>{len(painting_file_list)} paintings", painting_file_list)

    #for painting in painting_file_list:
    #LuisV:
    for painting in tqdm(painting_file_list):
    

        if args.verbose:
            print("    Handling " + painting)
        """
        interpretation_file_list = [y for x in
                                    [glob(interpretation_folder + '/%s*.%s' % (os.path.basename(painting), ext))
                                     for ext in image_extensions]
                                    for y in x]
        """
        #LuisV: support for subfolders
        interpretation_file_list = list()
        for (dirpath, dirnames, filenames) in os.walk(interpretation_folder):
            interpretation_file_list += [os.path.join(dirpath, file) for file in filenames]

        #LuisV
        #print(interpretation_file_list)
        #trace_log[painting]["colors_final"] = dict()
        #trace_log[painting]["colors_painting"] = dict()
        #trace_log[painting]["colors_background"] = dict()
        
        for interpretation in interpretation_file_list:
            ##
            # The following function will apply a style transfer on 'interpretation' as to have it
            # adopt the same style as 'painting'
            # The result is stored in 'interpretation'
            ##
            if args.verbose:
                print(f'    Saving {interpretation}')

            #LuisV
            #print("HEEEEEEYY")
            #print("interpretation", get_color_names(numpy.array(PIL.Image.open(interpretation) ) ) )
            
            #LuisV
            #trace_log[painting]["colors_background"][interpretation] = get_color_names(numpy.array(PIL.Image.open(interpretation) ) )
            #trace_log[painting]["colors_painting"][interpretation]  = get_color_names(numpy.array(PIL.Image.open(painting) ) )
            
            #LuisV
            background_image_path = interpretation
            final_image = apply_style_transfer(interpretation, painting, interpretation, args.rescale)
            #trace_log[painting] += f'{get_color_names(final_image)}'
            #LuisV
            #trace_log[painting]["colors_final"][interpretation] = get_color_names(final_image)
            
            #LuisV
            #print("final", get_color_names(final_image))
            #print("painting", get_color_names(numpy.array(PIL.Image.open(painting) ) ) )
            #print("OOOOOHHH")

            if args.verbose:
                print(f'    Done saving {interpretation}')

            if args.watermark:
                if args.verbose:
                    print(f'    Adding watermark {args.watermark}')

                final_image = cv2.imread(interpretation, cv2.IMREAD_UNCHANGED)
                final_image = add_watermark(final_image, args.watermark)
                cv2.imwrite(interpretation, final_image)

    if args.trace_file:
        with open(args.trace_file, 'w') as f:
            #f.write(json.dumps(trace_log))
            #LuisV
            f.write(json.dumps(trace_log, indent= 2 ))
        
        # LuisV: create a csv file
        output_csv = str(args.trace_file).replace(".json", "") + ".csv"
        df = pd.DataFrame(columns=[
            'image_file', 'method',         # mashup data
            'painting_name', 'painting_contains',  # painting data
            'background_name', 'background_colors', # background data
            'painting_path', 'background_path', # paths
            ])
        for painting in trace_log.keys():
            for mash_up_dict in trace_log[painting]["mash_ups"]:
                row_dict = {
                    'image_file': mash_up_dict["mash_up_path"],
                    'method': mash_up_dict['method'],
                    'painting_name': trace_log[painting]["painting_name"],
                    'painting_path': trace_log[painting]["painting_path"],
                    'painting_contains': trace_log[painting]["painting_contains"],
                    'background_name': mash_up_dict['background_name'],
                    'background_path': mash_up_dict['background_path'],
                    'background_colors': mash_up_dict['background_colors'],
                }

                df = df.append(row_dict, ignore_index= True)
        
        #save csv
        df.to_csv(output_csv)

                


if __name__ == "__main__":

    ##
    #  Input argument parsing declarations :
    #   - input_dir (-in)
    #   - substitute_dir (-sub)
    #   - output_dir (-out)
    #   - demo flag
    #   - nogui flag
    ##
    parser = argparse.ArgumentParser(prog="LaMuse",
                                     description='Generates reinterpretations of paintings')
    parser.add_argument("--input_dir", "-in", metavar='in', type=str, nargs=1,
                        help='input directory containing paintings to reinterpret (defaults to "'
                             + default_painting_folder + '" if non specified)', default=[default_painting_folder])
    parser.add_argument("--output_dir", "-out", metavar='out', type=str, nargs=1,
                        help='output directory for interpreted paintings (defaults to "' +
                             default_interpretation_folder + '" if non specified)',
                        default=[default_interpretation_folder])
    parser.add_argument("--background_dir", "-bck", metavar='bck', type=str, nargs=1,
                        help='background directory (defaults to "' +
                             default_background_folder + '" if non specified)',
                        default=[default_background_folder])
    parser.add_argument("--substitute_dir", "-sub", metavar='sub', type=str, nargs=1,
                        help='directory containing painting substitute images (defaults to "' +
                             default_image_folder + segmentation_suffix + '" if non specified)',
                        default=[default_image_folder + segmentation_suffix])
    parser.add_argument("--demo", action='store_true', help='Run in demo mode, reducing features to bare minimum')
    parser.add_argument("--nogui", action='store_true', help='Run in no-gui mode')
    parser.add_argument("-bw", action='store_true', help='Add greyscale filter')
    parser.add_argument("--verbose", action='store_true', help='Display trace messages')
    parser.add_argument("--rescale", action='store_true', help='Remove rescaling before applying style transfer')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(version_number))
    parser.add_argument("--watermark", "-wm", type=str, nargs='?', const=default_watermark_file,
                        help='watermark file (defaults to "' + default_watermark_file + '" if non specified)')

    parser.add_argument("--trace_file", "-tr", type=str, nargs='?',
                        help=f'output file for tracing all operations and their parameters (defaults to "{default_trace_file}" if non specified)',
                        const=default_trace_file)

    args = parser.parse_args()

    # Reset default directories if adequate arguments are provided
    default_painting_folder = args.input_dir[0]
    default_interpretation_folder = args.output_dir[0]
    default_substitute_folder = args.substitute_dir[0]
    default_background_folder = args.background_dir[0]

    print(default_painting_folder)
    print(default_interpretation_folder)
    print(default_substitute_folder)
    print(default_background_folder)

    # @TODO properly include stuff using pkg_resources
    if not os.path.isfile(mask_rcnn_config_file):
        if not args.nogui:
            sg.Popup(
                'LaMuse ne peut pas fonctionner sans le fichier %s. Merci de lire la documentation.' % mask_rcnn_config_file,
                title='Erreur')
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), mask_rcnn_config_file)

    # generate_images(args.input_dir, args.output_dir if args.output_dir else args.input_dir)

    window = sg.Window('Génération d\'images', layout)

    if args.nogui:
        if args.verbose:
            print("Calling full_case_study")

        generate_full_case_study(default_painting_folder, default_substitute_folder, default_background_folder,
                                 default_interpretation_folder)

        if args.verbose:
            print("Done calling full_case_study")

    else:
        while not args.nogui:
            event, values = window.read()
            if event in (None, 'Cancel'):
                break
            if event == "Génération de substituts":
                if values[0]:
                    default_image_folder = values[0]
                if not args.demo:
                    if args.verbose:
                        print("Generating substitute images")
                    generate_images(default_image_folder, default_substitute_folder)
                    if args.verbose:
                        print("Done generating substitute images")
                else:
                    sg.popup('%s inaccessible en mode demo' % event)

            if event == "Créer un cas d'étude":
                if values[0]:
                    default_image_folder = values[0]
                if values[1]:
                    default_painting_folder = values[1]
                if values[2]:
                    default_background_folder = values[2]

                sg.Popup("La génération de cas d'étude a commencé, en fonction du nombre de peintures fournies ceci "
                         "peut prendre un certain temps", title="Création démarrée", non_blocking=True)

                if args.verbose:
                    print("Calling full_case_study")

                generate_full_case_study(default_painting_folder, default_substitute_folder, default_background_folder,
                                         default_interpretation_folder)

                if args.verbose:
                    print("Done calling full_case_study")

                sg.Popup("Les résultats sont disponibles dans %s" % default_interpretation_folder,
                         title="Cas d'études terminé")

        window.close()
