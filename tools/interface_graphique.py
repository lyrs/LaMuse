import PySimpleGUI as sg
from .generate_segmented_pictures import generate_images

sg.theme('DarkAmber')

layout = [[sg.Text('Choisir un dossier d\'images "actuelles" à traiter')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Choisir un dossier (vide) pour les images crées : ')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Veuillez confirmer le traitement d\'images actuelles (ne pas fermer cette fenëtre durant le traitement) : ')],
          [sg.Button("Confirmer la génération d'images")],

          [sg.Text('Génération d\'un cas d\'étude :')],
          [sg.Text('Choisir un dossier de peintures : ')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Choisir un dossier d\'images de fond : ')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Choisir un dossier d\'images actuelles : ')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Text('Choisir un dossier de destination pour les résultats : ')],
          [sg.Input(), sg.FolderBrowse(initial_folder=".")],
          [sg.Button("Créer un nouveau cas d'étude")]
          ]

window = sg.Window('Génération d\'images', layout)

while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    if event == "Confirmer la génération d'images":
        generate_images(values[0],values[1])

    if event == "Créer un nouveau cas d'étude":
        ()

window.close()