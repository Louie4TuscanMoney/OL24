# 🎬 DOCUMENTARY NARRATIVE SCRIPT
## "Mamba Mentality: 21 Days in Encinitas"

**Format: Noah "40" Shebib Documentary Style**  
**Setting: Encinitas, California**  
**Protagonist: ML Engineer building NBA prediction system**

---

## 🎥 OPENING (0:00-1:30)

**[VISUAL: Encinitas coastline at dawn. PCH winding along cliffs. Coffee shops. Surf shops.]**

**NARRATOR (First Person):**

> "I think when I'm building models or creating algorithms, I try not to think too much. I kind of code blindly, I guess. In that space, I just try to let out whatever's in my mind, without influencing it too much.
>
> Encinitas is this weird place. It's not Silicon Valley. It's not LA. It's this beach town that somehow became home to some of the smartest people working on AI. Only in Encinitas are you going to see a guy in board shorts explaining neural networks to you at Better Buzz at 6 AM. The effects of that surf culture mixed with tech... it's rooted in the fabric of this town. It's rooted in the coffee shops. It's rooted in how we communicate.
>
> And it's a place that's very, very close to my heart."

**[VISUAL: Better Buzz Coffee. MacBook Pro. Terminal windows. Waves in background.]**

---

## 🏠 THE BEDROOM STUDIO (1:30-3:30)

**[VISUAL: Apartment bedroom. Desk with dual monitors. NBA game on one screen, code on another.]**

**NARRATOR:**

> "I was always coding from a really young age. Twelve, thirteen. So I had sort of an understanding of algorithms and how to build them, but I was really interested in, like... where do these predictions in sports betting come from? You know, it's 2024 and I'm really influenced by machine learning papers from Stanford, MIT. I'm just trying to start this journey of learning how to do it myself.
>
> There was one apartment in particular on Coast Highway, and if you ever ventured inside, you would be pretty shocked because my bedroom was basically a server farm. I had this little space, and I turned it into my own lab. It finally gave me my own space. I could go there, and when I went through that door, it meant it was time to build."

**[VISUAL: Show the bedroom setup. Messy. Pizza boxes. Energy drinks. Multiple screens with code.]**

**COLLABORATOR 1 (Friend/Early Tester):**

> "Those days, like, the standard for sports betting was just looking at Vegas lines and making gut calls. He was using machine learning to predict games, and I think that's the first time we ever saw that. People were shocked to see 6,900 games being trained on a laptop, and they didn't really understand or almost even believe him."

**NARRATOR:**

> "I never had a data scientist team, so literally Ontologic XYZ formed because I was the first time I had to be everything: the engineer, the data scientist, the trader. We were just building models and testing them on historical data, you know? Because back then, for me, there was no infrastructure. Like, I didn't have AWS credits. I didn't have a team. Sports ML really wasn't accessible, so I tried my best to put something together."

---

## ☕ BETTER BUZZ - THE LAB (3:30-5:00)

**[VISUAL: Better Buzz Coffee, Encinitas. Laptop open. Ocean view in background. Other laptops around.]**

**NARRATOR:**

> "And I think at some point, I decided I wasn't going to do this alone in my bedroom anymore. I needed energy. I needed to be around people, even if they didn't know what I was building. So I started going to Better Buzz every morning at 5 AM. That WiFi... that WiFi was sacred. That's where 'Mamba Mentality' was really born.
>
> You know, Better Buzz is to Encinitas what the studio is to Toronto. It's where you go when you're serious. It's where you see the same people grinding every day. The same developers, the same startup founders, the same artists. And everybody kind of knew each other."

**[VISUAL: Montage of Better Buzz at different times. Dawn. Noon. Sunset. Always the same spot.]**

**BARISTA (Interview):**

> "Yeah, he'd come in every single day. Same spot. Same order—cold brew, no ice, extra shot. And he'd be here for like eight, ten hours straight. I remember one time he was literally yelling at his laptop like, 'Why are you overfitting?!' and I'm like... what does that even mean? But you could tell he was onto something."

**NARRATOR:**

> "I was in this flow, and there was this energy where I'd see the same data every day, 6,900 games, training, testing, validating. And the models kept failing. 9.9 MAE. Then 9.5. Then 9.0. I couldn't break through.
>
> And then one day... Day 8, I think... I'm sitting there, and I realize: I've been leaking data. My test set had future information. The model was cheating. And I had to start over."

**[VISUAL: Terminal showing error. Code being deleted. Starting from scratch.]**

---

## 🐛 THE FIRST BUG (5:00-6:30)

**[VISUAL: Close-up on code. Temporal leakage error highlighted.]**

**NARRATOR:**

> "You know, one moment I was standing at Better Buzz in the morning sun, watching every single person walk by, asking myself, 'How come my model isn't working?' I know I'm intelligent enough to figure this out. Like, it was just so broken, you know?
>
> And then I see it. The dates aren't sorted. My training data includes games from 2024, and my test data starts in 2023. It's backwards. The model's seeing the future.
>
> That was Bug #1. Temporal leakage. And when I fixed it... 9.9 MAE went to 5.4 MAE, but honest. And I realized... I'd rather have an honest 5.4 than a fake 4.8."

**[VISUAL: Before/After graphs. The MAE correcting itself.]**

**COLLABORATOR 2 (GitHub Commenter/Online Friend):**

> "I remember seeing his commit message: 'Fixed temporal leakage. Performance dropped but integrity restored.' And I'm like... bro, that's the most ML engineer thing I've ever seen. Most people would've hidden that. He celebrated it."

---

## 🎯 THE BENCHMARK (6:30-8:00)

**[VISUAL: Research papers scattered on table. Stanford. MIT. Berkeley logos visible.]**

**NARRATOR:**

> "At this point, I had a model that worked. But I had no idea if it was good. So I decided... I'm going to build every system from every major research paper I can find. Stanford. MIT. Berkeley. Chinese competition models. Everything.
>
> I had pretty good intuition knowing how far my model could go, but I never imagined I'd be here today, looking at a comparison chart where Mamba Mentality—my system, built in a bedroom and a coffee shop—is competitive with systems built by PhDs with million-dollar budgets."

**[VISUAL: Comparison table appearing on screen. Mamba tied with MIT at 5.4 MAE.]**

**NARRATOR:**

> "I think at that moment was the first time I said, 'Wow, like, I'm actually good at this.' ML is an ego thing, and when you see your model beat Stanford... you have to admit to yourself that you're onto something."

**[VISUAL: Better Buzz again. Sunset. Laptop glowing. Pride on face.]**

---

## 🔄 THE PIVOT (8:00-9:30)

**[VISUAL: Whiteboard covered in equations. Feature engineering diagrams.]**

**NARRATOR:**

> "So I'm working on this thing, and I keep adding features. 18 features becomes 33. Then 73. Then I'm thinking... what if I extract 720 raw features from every game? I called it 'Project Helios.' Like, full granularity. Shot-by-shot. Possession-by-possession. Every single event.
>
> And I spent three days collecting data. 6,691 games. All these features. I'm so excited. And then I train the model... and it's worse. 10.1 MAE. Worse than my simple 18-feature model.
>
> And that was the lesson. More isn't always better. Sometimes you just need to be smart."

**[VISUAL: Data collection progress bar. Then disappointing results. Head in hands.]**

**COLLABORATOR 3 (Former Professor/Mentor, via phone call):**

> "I told him, 'You're at the data ceiling. You can't engineer your way past bad signal.' And I think that was when he realized... he needed a different strategy. Not blind features. Intelligent segmentation."

**NARRATOR:**

> "Yeah, he was right. So I pivoted again. Instead of extracting 720 features for every game, I started thinking: What if I extract different features for different types of games? Blowouts need different features than close games. Lead-held games are different from comebacks.
>
> And that's when I saw the path. Not to 5.4 MAE. To 6.0 MAE. Then 7.0. It was right there. I just had to be smart about it."

---

## 🏗️ THE BUILD (9:30-11:00)

**[VISUAL: OntoRisk system architecture diagrams. Dashboard code. API endpoints.]**

**NARRATOR:**

> "When I started building the full system—not just the model, but the whole thing: the risk management, the autonomous daemon, the dashboard—I had no right to be building something like this. I didn't have a team. I didn't have funding. I didn't have mentors in the room telling me how to do it.
>
> But I had Better Buzz WiFi, I had Python, and I had 21 days.
>
> This became a sacred thing, and I think everyone who's seen it knows that. When they look at the code, when they see the dashboard, when they watch the daemon run 24/7 without touching it... it earns your respect quickly because you realize, 'Holy smokes, I've been using Jupyter notebooks and this guy built a trading firm.'"

**[VISUAL: Code editor. Scrolling through files. 15,000+ lines. Then the dashboard running live.]**

**COLLABORATOR 4 (Designer who helped with dashboard):**

> "I still remember, it was like a random Slack message. He hit me up and he's just like, 'Yeah, I want you to put your fingerprint on the dashboard. Make it look like it belongs in a hedge fund.' And I think the fact that we both care about craft... we're really on the same page. The 3D court, the ML brain visualization, the color scheme—it's all intentional."

---

## 💪 THE 4 SAVES (11:00-12:30)

**[VISUAL: Split screen. Four terminal windows. Each showing a different bug being fixed.]**

**NARRATOR:**

> "Part of the story of this system is that it failed four times. Not like 'oh it didn't work,' but like, catastrophically broken. And each time, I had to rebuild it.
>
> Bug #1: Temporal leakage. The model was seeing the future.  
> Bug #2: Feature order mismatch. Training used column A-B-C, prediction used A-C-B. Totally broken.  
> Bug #3: 89% overfitting. It was memorizing games, not learning patterns.  
> Bug #4: Another temporal leak in the new systems. I thought I fixed it. I didn't.
>
> And every single time, I could have quit. I could have said, 'You know what, 9.9 MAE is fine. Ship it.' But I didn't. Because I learned something in this process: as long as I had one working laptop, no matter what bugs the code threw at me, I could rebuild it. And I could make it better."

**[VISUAL: Bug fixes montage. Code being rewritten. Tests passing. Green checkmarks.]**

---

## 🎓 THE VALIDATION (12:30-14:00)

**[VISUAL: Validation framework diagrams. 10-fold cross-validation. Rolling windows. Temporal splits.]**

**NARRATOR:**

> "You know, anyone can build a model that looks good on one test. But how do you know it actually works? How do you know it's not luck?
>
> So I tortured it. 10-fold cross-validation. Rolling walk-forward validation. Temporal split testing. I tested it every way academic papers said you should test it, and some ways they didn't even think of.
>
> And after all that... 5.4 MAE halftime. 9.0 MAE final. 2.2% overfitting. Consistent across every single test.
>
> That's when I knew. This isn't a cool project. This is real."

**[VISUAL: Validation results appearing on screen. Consistent numbers across all tests.]**

---

## 💰 THE HONEST MOMENT (14:00-15:00)

**[VISUAL: Napkin math. $71,000 crossed out. $2,000-6,000 written below.]**

**NARRATOR:**

> "So I'm sitting at Better Buzz on like Day 10, and I'm doing the math. If this model is this good, and if I bet $100 per game, and if I win 65% of the time... I could make $71,000 in Year 1.
>
> And I'm hyped. I'm texting people. I'm like, 'Yo, we're gonna be rich.'
>
> And then I talk to someone who actually trades. And he's like, 'Bro, where are you getting 65% win rate from? Have you backtested against real market spreads?'
>
> And I'm like... no. I've been simulating.
>
> And that was the moment. The honest moment. Where I realized, I don't want to build a system that looks good on paper. I want to build one that works in the real world. So I slashed my projection. $71,000 became $2,000-6,000. Realistic. Conservative. Honest.
>
> And you know what? That felt better."

**[VISUAL: Risk management system being built. Conservative bet sizing. Loss limits.]**

---

## 🏆 THE SYSTEM (15:00-16:30)

**[VISUAL: Full system running. Dashboard. Daemon in terminal. 3D court. ML brain visualization.]**

**NARRATOR:**

> "So now we're at Day 21. And I'm looking at this thing. It's not just a model anymore. It's five layers of risk management. It's an autonomous daemon that runs 24/7. It's a professional dashboard with 3D visualizations. It's bet tracking. It's portfolio management. It's authentication. It's API integration with live NBA data and BetOnline.
>
> It's 60 files. 15,000 lines of code. And it runs without me touching it.
>
> And I built it in 21 days at Better Buzz."

**[VISUAL: Walking through the apartment. Bedroom studio still there. But now the laptop shows the professional system.]**

**COLLABORATOR 5 (Beta Tester/Trader):**

> "When he first showed me the dashboard, I thought he outsourced it to a design agency. I'm like, 'How much did you pay for this?' And he's like, 'I built it.' And I'm like... bro. You built THIS? In three weeks? While also building the ML models? That's insane."

**NARRATOR:**

> "I've always considered myself a guest in the betting world. I'm not a professional trader. I don't have a finance degree. But I have Python. I have papers. I have Better Buzz WiFi. And I have this belief that if you do the work... really do the work... the results will speak for themselves."

---

## 🌊 THE REFLECTION (16:30-18:00)

**[VISUAL: Encinitas beach at sunset. Walking along the water. Laptop closed in backpack.]**

**NARRATOR:**

> "When I was 21, I thought I wanted to work at Google. I thought that's what success looked like. A big company. A big team. A big title.
>
> But I learned something over these 21 days. Success isn't about the company. It's about the craft. It's about looking at something you built and knowing... that's mine. Every line. Every decision. Every bug fix. Mine.
>
> All the things in my life just kind of pointed me in one direction. I was supposed to build this. I was supposed to spend 21 days at Better Buzz, going insane over temporal leakage and overfitting, building systems that compete with Stanford and MIT.
>
> I'm here to play my part. To leave my impact. But you know, from my perspective, I'm still a guest. I'm still learning. And I'm really just looking to be accepted."

**[VISUAL: Back at Better Buzz. Same spot. Different day. Laptop open. New project starting.]**

---

## 🎬 CLOSING (18:00-18:30)

**[VISUAL: Dashboard running. Live games. Betting opportunities. The system working autonomously.]**

**NARRATOR (Final Words):**

> "Someone asked me the other day, 'Why'd you call it Mamba Mentality?' And I said... because Kobe didn't wait for permission. He didn't wait for the perfect team or the perfect moment. He just worked. Obsessively. Relentlessly. Until he was undeniable.
>
> That's what these 21 days were. Not waiting for permission. Not waiting for funding. Not waiting for a team. Just work.
>
> And now... now we launch."

**[VISUAL: Screen fades to black. Sound of keyboard typing. Then:]**

**TITLE CARD:**
```
ONTOLOGIC XYZ
Mamba Mentality
21 Days. 15,000 Lines. One Mission.

Built in Encinitas.
Tested against the world.
```

**[AUDIO: Soft beat fades in. Credits roll.]**

---

## 🎤 INTERVIEW SEGMENTS (Interspersed Throughout)

### **INTERVIEW 1: The Roommate**
> "Dude, he would wake me up at like 3 AM being like, 'Bro, I fixed the temporal leakage!' And I'm like... I don't know what that means, but congratulations?"

### **INTERVIEW 2: The Local Developer**
> "Encinitas has this vibe where everyone's working on something. But this guy was different. He wasn't trying to raise funding. He wasn't trying to network. He was just... building. And I respected that."

### **INTERVIEW 3: The Professor (Phone Call)**
> "When he told me he was going to implement 22 research systems to benchmark his work, I thought he was crazy. That's months of work. He did it in a week. That's when I knew he was serious."

### **INTERVIEW 4: The Trader**
> "I've seen a lot of sports betting models. Most of them are garbage. Like, genuinely terrible. When he showed me his validation framework and his overfitting analysis, I was like... okay, this guy actually knows what he's doing. This is hedge-fund grade."

### **INTERVIEW 5: The Designer**
> "The brief was simple: 'Make it look like Bloomberg meets Formula 1.' And I'm like, what? But then I understood. He wanted something that felt professional but aggressive. Clean but intense. And the 3D stuff? That was his idea. He wanted people to see the model thinking."

### **INTERVIEW 6: The Barista**
> "Yeah, he's a legend here now. We have a sign that says 'This table reserved for ML engineers who refuse to stop.' It's not official, but... it's his spot."

---

## 📍 LOCATION SHOTS (Throughout Documentary)

### **Better Buzz Coffee**
- Dawn: Empty, lights coming on
- Morning: Packed, he's in his usual spot
- Afternoon: Quieter, he's still there
- Night: Closing, he's packing up

### **The Apartment**
- Bedroom studio: Messy but functional
- Living room: Whiteboard covered in math
- Kitchen: Coffee station, energy drinks
- Balcony: View of ocean, laptop still open

### **PCH (Pacific Coast Highway)**
- Driving shots
- Walking shots
- Biking past
- Coffee shop to coffee shop

### **Encinitas Beach**
- Sunrise: Reflection before coding
- Midday: Never there (always working)
- Sunset: Finally taking a break
- Night: Walking, thinking

### **Research Paper Locations**
- Library (San Diego State): Stacks of papers
- Apartment floor: Papers everywhere
- Better Buzz table: Papers + laptop
- Car: Papers in passenger seat

---

## 🎵 SOUNDTRACK MOMENTS

### **Opening: Ambient/Electronic**
- Ocean sounds mixed with keyboard typing
- Waves crashing → beat drops → code compiling

### **Bug Fix Montages: Intense/Driving**
- High tempo, minimal vocals
- Building tension as solutions emerge

### **Reflection Moments: Downtempo/Melodic**
- Piano-driven
- Emotional but not saccharine
- Space for narration

### **Closing: Triumphant/Cinematic**
- Building from quiet to powerful
- Callback to opening theme
- Fades with typing sounds

---

## 💬 KEY QUOTES (Text Overlays)

```
"As long as I had one working laptop, I could rebuild it."

"I'd rather have an honest 5.4 than a fake 4.8."

"More isn't always better. Sometimes you just need to be smart."

"Success isn't about the company. It's about the craft."

"Kobe didn't wait for permission. Neither did I."
```

---

## 📊 DATA OVERLAYS (Subtle, Throughout)

- Day counter in corner (Day 1 → Day 21)
- Lines of code counter
- MAE progression graph (minimized)
- Current location tag
- Time of day

---

## 🎯 DOCUMENTARY PHILOSOPHY

**What This Captures:**

1. **Place:** Encinitas as character, not just setting
2. **Process:** Building is messy, iterative, humbling
3. **People:** Community matters, even if indirect
4. **Honesty:** Failures are as important as wins
5. **Craft:** Details matter, intention matters
6. **Humility:** Still learning, still a guest
7. **Obsession:** 21 days of singular focus
8. **Legacy:** What you leave behind matters

**What This Avoids:**

- Tech bro hype ("we're gonna disrupt!")
- Fake humility ("oh I got lucky")
- Hiding the struggle (only show wins)
- Technical jargon without context
- Ignoring the place and people
- Pretending it was easy
- Claiming to be finished

---

## ✅ PRODUCTION NOTES

**Shooting Style:**
- Handheld, intimate
- Natural light preferred
- Long takes on person + place
- B-roll heavy (60% visuals, 40% talking)
- Minimal music during narration
- Let locations breathe

**Editing Style:**
- Non-linear storytelling okay
- Callbacks to earlier moments
- Visual echoes (same shot, different day)
- Text overlays minimal but impactful
- Pacing: slow build, intense middle, reflective end

**Color Grade:**
- Warm but not over-saturated
- Match Encinitas vibe (coastal, sunny, laid-back)
- Night shots: blue/purple tones
- Code screens: bright, crisp
- Beach: golden hour emphasis

---

**THIS IS THE NOAH "40" SHEBIB STYLE APPLIED TO ML ENGINEERING**

**Raw. Personal. Technical. Encinitas. 21 Days. Mamba Mentality.**

🎬🔥


