import time
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

print(gym.envs.registry.keys())


# Impostazione dell'ambiente (versione aggiornata)
env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# Creazione del modello PPO
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./ppo_mspacman_tensorboard/")

# Addestramento e visualizzazione del gameplay in tempo reale
total_timesteps = 1000000
obs = env.reset()

# Ciclo per addestrare e visualizzare
for _ in range(total_timesteps // 2048):
    model.learn(total_timesteps=2048, reset_num_timesteps=False)

    # Loop per visualizzare l'episodio mentre l'agente viene addestrato
    done = False
    while not done:  # Continua finché l'episodio non è finito
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # Rendering del frame
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (600, 800))
        cv2.imshow('MsPacman', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True  # Termina l'episodio se 'q' viene premuto

        time.sleep(0.01)  # Aggiungiamo una pausa per rallentare il video

    env.close()  # Chiudiamo l'ambiente alla fine dell'episodio
    cv2.destroyAllWindows()  # Chiudiamo la finestra di visualizzazione

    obs = env.reset()  # Reimpostiamo l'ambiente per il prossimo episodio

# Salvataggio del modello finale (fuori dal ciclo di addestramento)
model.save("ppo_mspacman_trained")