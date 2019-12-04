import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/pi/home/camo/')

from camo_gui import app as application
application.secret_key = 'ThisIsMySecretKey'