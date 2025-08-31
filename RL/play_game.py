import io
#import keyboard
import sys, select

from pyboy import PyBoy
import numpy as np
pyboy = PyBoy("/home/crafter_zelda/Zelda-Link-s-awakening-agent/RL/game_state/Link's awakening.gb")
save_file = "/home/crafter_zelda/Zelda-Link-s-awakening-agent/RL/game_state/Link's awakening.gb.state"

try:
    with open(save_file, "rb") as f:
        pyboy.load_state(f)
except FileNotFoundError:
    print("No existing save file, starting new game")

last_save_state = False

for i in range(10000):
    pyboy.tick()
    """
    之前的方案keyboard在linux下不太好用，故用sys来实现
    """
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.readline().strip()
        if key == 'x':
            if not last_save_state: 
                with open(save_file, "wb") as f:
                    pyboy.save_state(f)
                print(f"✅ 游戏状态已保存至 {save_file}")
                last_save_state = True
    #if keyboard.is_pressed('x'):
        
            else:
                last_save_state = False

    if (i%200 == 0):
        print("###############################")
        frame = pyboy.game_area()   # 或者 screen_image().convert('RGB') -> np.array
        #print(type(frame))       # 查看数据类型（应该是 numpy.ndarray）
        print(frame.shape)       # 数组形状，例如 (144, 160, 3)
        #print(frame.dtype)
        np.set_printoptions(threshold=np.inf)  # 关闭省略，打印完整数组
        print(frame)
        #print(pyboy.game_area())
        #print(sprite)
        #print (pyboy.memory[0xDBAE])
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.readline().strip()
        if key == 'q':
            break

pyboy.stop()
