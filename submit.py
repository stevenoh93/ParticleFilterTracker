import time
import os
import sys
import argparse
import json
import datetime
from bonnie.submission import Submission

def main():
  parser = argparse.ArgumentParser(description='Submits code to the Udacity site.')
  parser.add_argument('--provider', choices = ['gt', 'udacity'], default = 'udacity')
  parser.add_argument('--environment', choices = ['local', 'development', 'staging', 'production'], default = 'production')

  args = parser.parse_args()
  quiz = 'ps07'

  submission = Submission('cs6476', quiz, 
                          filenames = ["ps7.py"],
                          environment = args.environment, 
                          provider = args.provider)

  timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())

  while not submission.poll():
    time.sleep(3.0)

  if submission.feedback():
    feedback = submission.result()

    filename = "%s-result-%s.json" % (quiz, timestamp)

    with open(filename, "w") as fd:
      json.dump(feedback, fd, indent=4, separators=(',', ': '))

    print feedback['console_summary']

    print "(Details available in %s.)" % filename

  elif submission.error_report():
    error_report = submission.error_report()
    print json.dumps(error_report, indent=4)
  else:
    print "Unknown error."

if __name__ == '__main__':
  main()
