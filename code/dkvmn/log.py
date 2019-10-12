import subprocess, logging, time, sys


def setlogger(logdir, log_name):
    subprocess.call(['mkdir', '-p', logdir])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    logfile = logging.FileHandler(logdir + "/" + log_name + timestr, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    try:
        GIT_BRANCH = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'])
    except:
        GIT_BRANCH = "Unknown"
    try:
        GIT_REVISION = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        GIT_REVISION = "Unknown"

    logger.info('COMMAND: [ nohup python %s & ], GIT_REVISION: [%s] [%s]'
                % (' '.join(sys.argv), GIT_BRANCH.strip(), GIT_REVISION.strip()))
    return logger