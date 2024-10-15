from random import randrange

done_filename = 'done.txt'
ignore_filename = 'ignore.txt'
TOTAL_NUM = 3000

with open(done_filename, 'r') as f:
	done_set = [int(line.strip()) for line in f.readlines() if line.strip() != '']

with open(ignore_filename, 'r') as f:
	ignore_set = [int(line.strip()) for line in f.readlines() if line.strip() != '']
	print(ignore_set)

while True:
	r = randrange(1, TOTAL_NUM)
	while r in done_set or r in ignore_set:
		r = randranfe(1, TOTAL_NUM)
	print(r)
	print('Page', r//50+1)
	print('https://leetcode.cn/problemset/?page={}'.format(r//50+1))
	choice = input('[y]es, [n]o, [i]gnore or [b]reak\n')
	if choice.strip() == 'y':
		with open(done_filename, 'a') as f:
			f.write('{}\n'.format(r))
			break
	elif choice.strip() == 'b':
		break
	elif choice.strip() == 'i':
		with open(ignore_filename, 'a') as f:
			f.write('{}\n'.format(r))
			continue
