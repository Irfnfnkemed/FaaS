import subprocess
from typing import List

from .ipc import ClientIPC, Server, get_free_port, get_ip


class Job:
    def __init__(self):
        self._ipc = ClientIPC()
        self._server = Server(get_free_port("localhost"))
        self._process_list: List[subprocess.Popen] = []

    def run(self, sched_ip: str, sched_port: int, script_path: str, args: List[str]):
        self._ipc.connect(sched_ip, sched_port)
        cmd, gpu_list = self._ipc.recv()
        assert cmd == 'alloc-to'
        if len(gpu_list) == 0:
            raise RuntimeError()
        self._server.serve()
        while True:
            parsed_list = {}
            for entry in gpu_list:
                ip, gpu_id = entry.split(':')
                if ip in parsed_list:
                    parsed_list[ip].append(gpu_id)
                else:
                    parsed_list[ip] = [gpu_id]
            parsed_gpu_list = [(ip, ','.join(ids)) for ip, ids in parsed_list.items()]
            master_port = get_free_port(parsed_gpu_list[0][0])
            proxy_ip = get_ip()
            for index, (ip, gpus) in enumerate(parsed_gpu_list):
                job_cmd = ["ssh", f"{ip}",
                           f"CUDA_VISIBLE_DEVICES={gpus}",
                           "torchrun", script_path,
                           f"--nproc_per_node={len(gpus)}",
                           f"--nnodes={len(parsed_gpu_list)}",
                           f"--node_rank={index}",
                           f"--master_addr={parsed_gpu_list[0][0]}",
                           f"--master_port={master_port}",
                           f"--proxy_ip={proxy_ip}",
                           f"--proxy_port={self._server.get_port()}"
                           ] + args
                self._process_list.append(subprocess.Popen(
                    job_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True))
            server_ipc = self._server.accept()  # accept conn from rank0
            while True:
                cmd, data = server_ipc.recv()
                if cmd == 'end':
                    for process in self._process_list:
                        stdout, stderr = process.communicate()
                        if process.returncode != 0:
                            print(stderr)
                    self._server.close()
                    server_ipc.close()
                    self._ipc.send('end', '')
                    self._ipc.close()
                    return
                elif cmd == 'alloc':
                    gpu_num = int(data)
                    self._ipc.send('alloc', gpu_num)
                    cmd_response, new_gpu_list = self._ipc.recv()
                    assert cmd_response == 'alloc'
                    server_ipc.send('alloc', len(new_gpu_list))
                    if len(new_gpu_list) > 0:  # save checkpoint and restart
                        for process in self._process_list:
                            stdout, stderr = process.communicate()
                            if process.returncode != 0:
                                print(stderr)
                        server_ipc.close()
                        gpu_list = new_gpu_list
                        break
