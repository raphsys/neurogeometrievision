import os
import ast

output_filename = "suvivi_03012026_COMPLET.txt"
project_dirs = ["neurogeomvision", "examples"]
files_to_include = ["setup.py", "requirements.txt"]

def get_file_description(file_path):
    """Tente d'extraire la docstring du début du fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        return ast.get_docstring(tree) or "Aucune description disponible."
    except Exception:
        return "Impossible d'analyser la description (erreur de parsing ou fichier non-Python)."

def write_separator(f, title, char="="):
    f.write(f"\n{char * 80}\n")
    f.write(f"{title}\n")
    f.write(f"{char * 80}\n\n")

with open(output_filename, 'w', encoding='utf-8') as outfile:
    # En-tête du fichier global
    outfile.write(f"# SNAPSHOT COMPLET DU PROJET NEUROGEOMVISION\n")
    outfile.write(f"# Date: 03 Janvier 2026\n")
    outfile.write(f"# Ce fichier contient l'intégralité du code source et des tests.\n\n")

    # 1. Fichiers à la racine
    for fname in files_to_include:
        if os.path.exists(fname):
            write_separator(outfile, f"FICHIER RACINE : {fname}")
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + "\n")
            except Exception as e:
                outfile.write(f"Erreur de lecture : {e}\n")

    # 2. Parcours des répertoires du projet
    for root_dir in project_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Ignorer __pycache__
            if "__pycache__" in dirpath:
                continue
                
            for filename in sorted(filenames):
                if filename.endswith(".py") or filename.endswith(".md"):
                    file_path = os.path.join(dirpath, filename)
                    
                    # Extraction description pour les .py
                    description = ""
                    if filename.endswith(".py"):
                        description = get_file_description(file_path)
                    
                    write_separator(outfile, f"FICHIER : {file_path}")
                    
                    if description:
                        outfile.write(f"""\nDESCRIPTION :\n{description}\n"""\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read() + "\n")
                    except Exception as e:
                        outfile.write(f"Erreur de lecture : {e}\n")

print(f"Fichier complet généré : {output_filename}")
