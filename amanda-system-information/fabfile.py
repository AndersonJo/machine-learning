# -*- coding:utf-8 -*-
import os

from fabric.api import *
# Force with you
from awsfabrictasks import default_settings as settings
from awsfabrictasks.decorators import ec2instance
from awsfabrictasks.conf import Settings

# env.key_filename = ['~/.ssh/fission.pem']
# env.user = 'ubuntu'
BASE_DIR = os.path.dirname(__file__)
env.key_filename = ['~/.ssh/fission.pem', '~/.ssh/amanda.pem']
env.user = 'ubuntu'
env.skip_bad_hosts = True


@task
@ec2instance(instanceid='i-c79faa62')
def hello():
    settings = Settings()
    print settings.as_dict()
    run('ls -ls')
