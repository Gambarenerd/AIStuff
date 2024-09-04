import time
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

### Impostazione dell'ambiente
env = make_atari_env('MsPacmanDeterministic-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

### Creazione del modello PPO
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_mspacman_tensorboard/")

### Addestramento e visualizzazione del gameplay in tempo reale
total_timesteps = 1000000
obs = env.reset()
num_episodes = 0  # Contatore per il numero di episodi

# Ciclo per addestrare e visualizzare
for _ in range(total_timesteps // 2048):
    model.learn(total_timesteps=2048, reset_num_timesteps=False)

    # Loop per visualizzare l'episodio mentre l'agente viene addestrato
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # Rendering del frame
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (600, 800))
        cv2.imshow('MsPacman', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if done.any():
            obs = env.reset()
            num_episodes += 1  # Incrementa il contatore di episodi
            print(f"Episodi completati: {num_episodes}")

        time.sleep(0.05)

env.close()
cv2.destroyAllWindows()
