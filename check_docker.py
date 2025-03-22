# check_docker.py
import subprocess
import sys

def check_docker():
    try:
        # Verifica se Docker è installato
        docker_version = subprocess.run(["docker", "--version"], 
                                        capture_output=True, 
                                        text=True,
                                        check=True)
        print(f"Docker installato: {docker_version.stdout.strip()}")
        
        # Verifica se Docker è in esecuzione
        docker_info = subprocess.run(["docker", "info"], 
                                    capture_output=True, 
                                    text=True)
        
        if docker_info.returncode == 0:
            print("Docker è in esecuzione.")
        else:
            print("Docker non è in esecuzione. Avvia Docker Desktop.")
            sys.exit(1)
        
        # Verifica Docker Compose
        compose_version = subprocess.run(["docker-compose", "--version"], 
                                        capture_output=True, 
                                        text=True,
                                        check=True)
        print(f"Docker Compose installato: {compose_version.stdout.strip()}")
        
        return True
    except subprocess.CalledProcessError:
        print("Docker o Docker Compose non sono installati correttamente.")
        return False
    except FileNotFoundError:
        print("Docker o Docker Compose non sono installati o non sono nel PATH.")
        return False

if __name__ == "__main__":
    if check_docker():
        print("\nDocker è configurato correttamente!")
    else:
        print("\nC'è un problema con l'installazione di Docker.")
        print("Scarica Docker Desktop da: https://www.docker.com/products/docker-desktop")