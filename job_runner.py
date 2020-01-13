import schedule
import time
from Predicter import Predicter

schedule.every(10).minutes.do(Predicter.predict)

while True:
    schedule.run_pending()
    time.sleep(1)
