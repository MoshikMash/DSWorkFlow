from shutil import copyfile
from threading import Timer
import os
from time import gmtime, strftime
import filecmp

participant_name = 'INSERT PARTICIPANT NAME'
is_tutorial = False

os.chdir('INSERT PATH')

TUTORIAL_SNAPSHOTS_PATH = 'SNAPSHOTS/{}/{}/'.format(participant_name, 'tutorial')
EXPERIMENT_SNAPSHOTS_PATH = 'SNAPSHOTS/{}/{}/'.format(participant_name, 'experiment')

source_file = None
TARGET_PATH = None


def Counter(increase_the_counter=True):
    if 'cnt' not in Counter.__dict__:
        Counter.cnt = 0
    if increase_the_counter:
        Counter.cnt += 1
    return Counter.cnt


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def hello():
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(':', '-'))


def copy_file():
    counter = Counter(increase_the_counter=False)
    #    time = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(':','-')

    target_file = os.path.join(TARGET_PATH, '{}_{}.ipynb'.format(participant_name, counter))
    if os.path.exists(target_file):
        if not filecmp.cmp(target_file, source_file):
            counter = Counter(increase_the_counter=True)
            target_file = os.path.join(TARGET_PATH, '{}_{}.ipynb'.format(participant_name, counter))
            copyfile(source_file, target_file)
            print('{}_{}.ipynb was created'.format(participant_name, counter))
        else:
            print('There is no difference between the files')
    else:
        counter = Counter(increase_the_counter=True)
        target_file = os.path.join(TARGET_PATH, '{}_{}.ipynb'.format(participant_name, counter))
        copyfile(source_file, target_file)
        print('{}_{}.ipynb was created'.format(participant_name, counter))


if not os.path.exists(TUTORIAL_SNAPSHOTS_PATH):
    os.makedirs(TUTORIAL_SNAPSHOTS_PATH)

if not os.path.exists(EXPERIMENT_SNAPSHOTS_PATH):
    os.makedirs(EXPERIMENT_SNAPSHOTS_PATH)

if is_tutorial:
    source_file = 'INSERT THE TUTORIAL SOURCE FILE NAME (NOTEBOOK FILE)'
    TARGET_PATH = TUTORIAL_SNAPSHOTS_PATH
else:
    source_file = 'INSERT THE EXPERIMENT SOURCE FILE NAME (NOTEBOOK FILE)'
    TARGET_PATH = EXPERIMENT_SNAPSHOTS_PATH
print("starting...")
rt = RepeatedTimer(1, copy_file)  # it auto-starts, no need of rt.start()


# rt.stop()






