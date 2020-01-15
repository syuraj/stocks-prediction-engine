import schedule
import time
from Predictor import Predictor

schedule.every(10).minutes.do(Predictor.predict)

while True:
    schedule.run_pending()
    time.sleep(1)
