import sys
from glob import glob
import os
import errno
import argparse
import pkg_resources

import PySimpleGUI as sg
from .tools.generate_segmented_pictures import generate_images
from .tools.create_original_case_study import create_case_study
from .tools.fast_style_transfer import save_image

import cv2

from .tools.watermarking import add_watermark

segmentation_suffix = "_objets"

# @Todo: Currently configuration data is packed with the software and stored in the /bin or /lib
#   directory after installation/deployment. This should be changed to a more convenient location
default_image_folder = f'{os.path.dirname(__file__)}/BaseImages'
default_background_folder = f'{os.path.dirname(__file__)}/Backgrounds'
default_painting_folder = f'{os.path.dirname(__file__)}/Paintings'
default_interpretation_folder = './Interpretations'
default_watermark_file = f'{os.path.dirname(__file__)}/Watermark.png'

mask_rcnn_config_file = f'{os.path.dirname(__file__)}/mask_rcnn_coco.h5'

version_number = '0.1.0'

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
    # The following function will go over all images in 'default_painting_folder' and use
    # the Mask_RCNN neural network to find identifiable objects.
    # It will then substitute these objects with similar ones stored in the 'default_substitute_folder'
    # folder, and replace the background with a random image chosen from 'default_background_folder'
    # The results are stored in 'dafault_interpretation_folder'
    ##
    if args.verbose:
        print("   Calling create_case_study")

    create_case_study(painting_folder, substitute_folder,
                      background_folder, interpretation_folder, 1)

    if args.verbose:
        print("   Done calling create_case_study")

    ##
    # Go over all images in 'default_painting_folder' and the corresponding images in
    # 'default_interpretation_folder' and apply a style transfer.
    ##
    image_extensions = ["jpg", "gif", "png", "tga", "jpeg"]
    painting_file_list = [y for x in [glob(painting_folder + '/*.%s' % ext) for ext in image_extensions]
                          for y in x]

    for painting in painting_file_list:

        if args.verbose:
            print("    Handling " + painting)

        interpretation_file_list = [y for x in
                                    [glob(interpretation_folder + '/%s*.%s' % (os.path.basename(painting), ext))
                                     for ext in image_extensions]
                                    for y in x]
        for interpretation in interpretation_file_list:
            ##
            # The following function will apply a style transfer on 'interpretation' as to have it
            # adopt the same style as 'painting'
            # The result is stored in 'interpretation'
            ##

            if args.verbose:
                print(f'    Saving {interpretation}')

            save_image(interpretation, painting, interpretation)

            if args.verbose:
                print(f'    Done saving {interpretation}')

            if args.watermark:
                if args.verbose:
                    print(f'    Adding watermark {args.watermark}')

                image = cv2.imread(interpretation, cv2.IMREAD_UNCHANGED)
                image = add_watermark(image, args.watermark)
                cv2.imwrite(interpretation, image)


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
    parser.add_argument("--verbose", action='store_true', help='Display trace messages')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(version_number))
    parser.add_argument("--watermark", "-wm", type=str, nargs='?', const=default_watermark_file,
                        help='watermark file (defaults to "' + default_watermark_file + '" if non specified)')

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
                else:
                    default_background_folder = default_image_folder + '/Backgrounds'

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
