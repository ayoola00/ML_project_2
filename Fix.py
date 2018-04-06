inp = open('sample.pickle', 'rb')
str_inp = inp.read().decode()
modified_file = str_inp.replace('\r\n', '\n')
inp.close()

out = open('sample.pickle', 'wb')
out.write(modified_file.encode())
out.close()
