import os
import shutil

root_dir = "/ruta/a/tu/directorio"

for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    letra_dir = os.path.join(root_dir, letra)
    if os.path.isdir(letra_dir):
        sample_dir = os.path.join(letra_dir, "sample")
        os.makedirs(sample_dir, exist_ok=True)

        for file_name in os.listdir(letra_dir):
            if file_name.endswith(".jpg"):
                src_path = os.path.join(letra_dir, file_name)
                dest_path = os.path.join(sample_dir, file_name)
                shutil.move(src_path, dest_path)

        print(f"Archivos movidos a {sample_dir} para la letra {letra}.")
    else:
        print(f"La carpeta {letra} no existe.")

print("Proceso completado.")
