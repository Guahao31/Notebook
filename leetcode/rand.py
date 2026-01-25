from random import randrange

done_filename = 'done.txt'
ignore_filename = 'ignore.txt'
todo_filename = 'todo.txt'
TOTAL_NUM = 3000

with open(done_filename, 'r') as f:
	done_set = [int(line.strip()) for line in f.readlines() if line.strip() != '']

with open(ignore_filename, 'r') as f:
	ignore_set = [int(line.strip()) for line in f.readlines() if line.strip() != '']
 
with open(todo_filename, 'r') as f:
	todo_set = [int(line.strip()) for line in f.readlines() if line.strip() != '']

while True:
	r = randrange(1, TOTAL_NUM)
	while r in done_set or r in ignore_set or r in todo_set:
		r = randrange(1, TOTAL_NUM)
	print(r)
	print('https://leetcode.cn/search/?q={}'.format(r)
	choice = input('[y]es, [n]o, [i]gnore, [t]odo or [b]reak\n')
	if choice.strip() == 'y':
		with open(done_filename, 'a') as f:
			f.write('{}\n'.format(r))
			break
	elif choice.strip() == 'i':
		with open(ignore_filename, 'a') as f:
			f.write('{}\n'.format(r))
			continue
	elif choice.strip() == 't':
		with open(todo_filename, 'a') as f:
			f.write('{}\n'.format(r))
			continue
	elif choice.strip() == 'b':
		break
	
