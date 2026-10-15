"""Launch from a normal user terminal; verify readiness before opening browser."""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT=Path(os.path.abspath(__file__)).parent


def ensure_frontend():
    dist=ROOT/'frontend/dist/index.html'
    if dist.exists(): return
    npm=shutil.which('npm.cmd' if os.name=='nt' else 'npm')
    if not npm:
        raise RuntimeError('Frontend build bulunamadı. Node.js LTS kurun veya build içeren ZIP paketini kullanın.')
    print('React arayuzu kuruluyor ve derleniyor…',flush=True)
    subprocess.run([npm,'ci','--no-audit','--no-fund'],cwd=ROOT/'frontend',check=True)
    subprocess.run([npm,'run','build'],cwd=ROOT/'frontend',check=True)


def main():
    os.chdir(ROOT)
    ensure_frontend()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    log_path=ROOT/'baslatma.log'
    with log_path.open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port',str(port)],stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                if process.poll() is not None: raise RuntimeError('Sunucu baslatilamadi: '+str(log_path))
                try:
                    with opener.open(url+'/api/health',timeout=1) as response:
                        if response.status==200: break
                except Exception: time.sleep(.5)
            else: raise RuntimeError('Sunucu yanit vermedi: '+str(log_path))
            print('\nYorum Radar hazir: '+url,flush=True)
            print('Bu pencereyi acik tutun. Durdurmak icin Ctrl+C.',flush=True)
            webbrowser.open(url)
            process.wait()
        finally:
            if process.poll() is None:
                process.terminate();process.wait()


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: pass
    except Exception as error:
        print('\nBaslatma hatasi: '+str(error),flush=True)
        sys.exit(1)
