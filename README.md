# csi-sensing
Channel State Information Sensing in the Wi-Fi Network


## Clone this repository (with submodules)

To clone this project including all submodules, use:

```bash
git submodule update --init --recursive
```

In the collect-csi-data, you can edit the prvInitialiseNewTask tasks.c:1111 (uxPriority < 25)

```c
xTaskCreatePinnedToCore(&vTask_socket_transmitter_sta_loop,
                        "socket_transmitter_sta_loop",
                        10000,
                        (void *)&is_wifi_connected,
                        20,
                        &xHandle,
                        1);
```
