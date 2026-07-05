#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

PATCH_PATH="${1:-/tmp/vllm-ray-v2-tcpstore-port.patch}"
VLLM_ACTOR_PYTHON="${VLLM_ACTOR_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
VLLM_SITE_PACKAGES=$("${VLLM_ACTOR_PYTHON}" -c \
    'import pathlib, vllm.v1.executor.ray_executor_v2 as m; print(pathlib.Path(m.__file__).resolve().parents[3])')
VLLM_EXECUTOR_PATH="${VLLM_SITE_PACKAGES}/vllm/v1/executor/ray_executor_v2.py"
VLLM_MQ_PATH="${VLLM_SITE_PACKAGES}/vllm/distributed/device_communicators/shm_broadcast.py"

if git -C "${VLLM_SITE_PACKAGES}" apply --check "${PATCH_PATH}"; then
    git -C "${VLLM_SITE_PACKAGES}" apply "${PATCH_PATH}"
elif git -C "${VLLM_SITE_PACKAGES}" apply --reverse --check "${PATCH_PATH}"; then
    echo "vLLM Ray V2 TCPStore patch is already applied"
else
    git -C "${VLLM_SITE_PACKAGES}" apply --check "${PATCH_PATH}"
fi

grep -F 'def _select_tcpstore_port()' "${VLLM_EXECUTOR_PATH}"
grep -F 'bind_to_random_port' "${VLLM_MQ_PATH}"
"${VLLM_ACTOR_PYTHON}" -m py_compile "${VLLM_EXECUTOR_PATH}" "${VLLM_MQ_PATH}"
VLLM_PORT=20001 "${VLLM_ACTOR_PYTHON}" -c '
from concurrent.futures import ThreadPoolExecutor

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils.network_utils import get_open_port
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

tcpstore_port = RayExecutorV2._select_tcpstore_port()
mq_port = get_open_port()
assert 20065 <= tcpstore_port <= 20096, tcpstore_port
assert mq_port == 20001, mq_port
assert tcpstore_port != mq_port

with ThreadPoolExecutor(max_workers=8) as executor:
    queues = list(
        executor.map(
            lambda _: MessageQueue(1, 0, connect_ip="127.0.0.1"),
            range(8),
        )
    )
addresses = [queue.handle.remote_subscribe_addr for queue in queues]
ports = [int(address.rsplit(":", 1)[1]) for address in addresses]
assert len(set(ports)) == len(ports), ports
assert all(20001 <= port < 20065 for port in ports), ports
for queue in queues:
    queue.remote_socket.close(linger=0)
'
