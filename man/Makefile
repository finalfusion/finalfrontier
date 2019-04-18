all: ff-train-skipgram.1 ff-train-deps.1

clean:
	rm -f *.1 *.5

%: %.md
	pandoc -s -w man -o $@ $<
