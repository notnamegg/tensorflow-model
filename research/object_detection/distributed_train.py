import json
import os
import tensorflow as tf
import train
import time
import subprocess
import sys,traceback
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('ps', '', 'host of pss.')
flags.DEFINE_string('worker', '', 'host of workers.')
flags.DEFINE_string('worker2nd', '', 'host of second workers for run 2 worker at one host.')
flags.DEFINE_integer('id', 0, 'index of job.')
flags.DEFINE_string('type', '', 'type of job.')
flags.DEFINE_string('masterport', '3000', 'port of master.')
flags.DEFINE_string('psport', '3001', 'port of ps.')
flags.DEFINE_string('workerport', '3002', 'port of worker.')
flags.DEFINE_string('workerport2nd', '3003', 'port of second worker.')
#flags.DEFINE_string('pipeline_config_path', 'object_detection/protos/ssd_mobilenet_v1_coco.ptoto',
#                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                    'file. If provided, other configs are ignored')
#flags.DEFINE_string('train_dir', '/tmp/train_log', 'Path to a train_pb2.TrainConfig config file.')
FLAGS = flags.FLAGS
def main(_):
	assert FLAGS.type, '`type` is missing.'
	assert FLAGS.ps, '`ps` is missing.'
	assert FLAGS.master, '`master` is missing.'
	if FLAGS.type == "master":
		assert FLAGS.train_dir, '`train_dir` is missing.'
		assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing.'
	# tf_config=json.loads('{'cluster': {'master': ["$master:3000"], 'ps': ["$ps:3001"], 'worker': ["$worker:3002","$worker2:3002"]}, 'task': {'index':0, 'type': "$job"}}')
	mslst=FLAGS.master.split(',')
	master=mslst[0]
	ps=FLAGS.ps
	pslst=FLAGS.ps.split(',')
	# ps='192.168.1.46'
	wklst=FLAGS.worker.split(',')
	wklst2nd=FLAGS.worker2nd.split(',')
	# worker='192.168.1.52'
	tf_config={}
	cluster={}
	masterlist=[]
	masterlist.append(master+":"+FLAGS.masterport)
	cluster['master']=masterlist
	pslist=[]
	for p in pslst:
		pslist.append(p+":"+FLAGS.psport)
	cluster['ps']=pslist
	workerlist=[]
	for w in wklst:
		if w != "" :
			workerlist.append(w+":"+FLAGS.workerport)
	for w2 in wklst2nd:
		if w2 != "":
			workerlist.append(w2+":"+FLAGS.workerport2nd)
	if len(workerlist) != 0:
		cluster['worker']=workerlist
	tf_config['cluster']=cluster
	task={}
	task['index']=FLAGS.id
	if FLAGS.type=="worker2nd":
		task['type']="worker"
	else:
		task['type']=FLAGS.type
	tf_config['task']=task
	os.environ['TF_CONFIG']=json.JSONEncoder().encode(tf_config)
	print(json.loads(os.environ.get('TF_CONFIG', '{}')))
	FLAGS.master='master'
	FLAGS.task=0
	retry=0
	while True:
		try:
			print("retry = "+str(retry))
			my_env = os.environ.copy()
			myprog = sys.argv[0].replace("distributed_train.py","train.py")
			cmd="python "+myprog+" --pipeline_config_path="+FLAGS.pipeline_config_path+" --train_dir="+FLAGS.train_dir
			myargv = sys.argv
			print(cmd)
			proc = subprocess.Popen(cmd, shell=True,env=my_env)
			(out, err) = proc.communicate()
			if proc.returncode == 0:
				break
			else:
				print("Error: "+str(proc.returncode))
				retry=retry+1
				time.sleep(30)
			# train.main(_)
		except KeyboardInterrupt:
			print("Ctrl c END")
			break
		except:
			tb = traceback.format_exc()
			print(tb)
			retry=retry+1
			time.sleep(30)
		
	# os.system("python object_detection/train.py --master=master --task=0")
	# os.system("python object_detection/train.py --master=master --task=0 --logtostderr --pipeline_config_path=object_detection/protos/ssd_mobilenet_v1_coco.ptoto  --train_dir=/tmp/train_log")

if __name__ == '__main__':
  tf.app.run()
