all: ff-convert.1 ff-similar.1 ff-train.1 ff-format.5

clean:
	rm -f *.1 *.5

%: %.md
	pandoc -s -w man -o $@ $<
