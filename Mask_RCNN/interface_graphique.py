import PySimpleGUI as sg

sg.theme('DarkAmber')

layout = [[sg.Text('Choisir un dossier d\'images "actuelles" à traiter')],
          [sg.Input(), sg.FolderBrowse()],
          [sg.Text('Choisir un nom de dossier pour les images crées : ')],
          [sg.Input()],
          [sg.Text('Veuillez confirmer le traitement d\'images actuelles (ne pas fermer cette fenëtre durant le traitement) : ')],
          [sg.Button("Confirmer la génération d'images")]
          ]

window = sg.Window('Window Title', layout)

while True:
    event, values = window.read()
    if event in (None, 'Cancel'):
        break
    if event == "Confirmer la génération d'images":
        print("ok.")
    print('You entered ', values[0])

window.close()