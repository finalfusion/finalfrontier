all: finalfrontier.1 finalfrontier-skipgram.1 finalfrontier-deps.1

clean:
	rm -f *.1 *.5

%: %.md
	pandoc -s -w man -o $@ $<
