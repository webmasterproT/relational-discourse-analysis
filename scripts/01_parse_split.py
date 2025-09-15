#!/usr/bin/env python3
"""
Parse and segment conversation data by speaker.
Outputs structured JSON conforming to utterance.schema.json
"""

import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RawUtterance:
    speaker: str
    text: str
    timestamp: Optional[str] = None
    context_markers: Optional[List[str]] = None

class ConversationParser:
    def __init__(self):
        self.speaker_patterns = [
            (r'^Person [Aa]\s*:?\s*(.*)$', 'PERSON_A'),
            (r'^Person [Bb]\s*:?\s*(.*)$', 'PERSON_B'),
            (r'^person [ab]\s*:?\s*(.*)$', 'PERSON_A' if 'a' in r'^person [ab]' else 'PERSON_B'),
            (r'^\s*[Aa]\s*[:.]?\s*(.*)$', 'PERSON_A'),
            (r'^\s*[Bb]\s*[:.]?\s*(.*)$', 'PERSON_B'),
        ]
        
        self.timestamp_pattern = r'(\d{1,2}:\d{2}(?:am|pm)?)'
        
        # Contextual markers for special sections
        self.context_markers = [
            'call', 'missed calls', 'conversation', 'message', 
            'hospital', 'mental institution', 'court', 'appointment'
        ]
    
    def detect_speaker(self, line: str) -> Tuple[Optional[str], str]:
        """Detect speaker and extract utterance text."""
        line = line.strip()
        if not line:
            return None, ""
            
        for pattern, speaker in self.speaker_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                return speaker, text
        
        return None, line
    
    def extract_timestamp(self, text: str) -> Tuple[Optional[str], str]:
        """Extract timestamp if present."""
        match = re.search(self.timestamp_pattern, text)
        if match:
            timestamp = match.group(1)
            # Remove timestamp from text
            cleaned_text = re.sub(self.timestamp_pattern, '', text).strip()
            return timestamp, cleaned_text
        return None, text
    
    def parse_conversation_text(self, text: str) -> List[RawUtterance]:
        """Parse raw conversation text into structured utterances."""
        lines = text.split('\n')
        utterances = []
        current_speaker = None
        current_text_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for speaker change
            detected_speaker, text_content = self.detect_speaker(line)
            
            if detected_speaker:
                # Save previous utterance if exists
                if current_speaker and current_text_parts:
                    combined_text = ' '.join(current_text_parts)
                    timestamp, cleaned_text = self.extract_timestamp(combined_text)
                    
                    utterances.append(RawUtterance(
                        speaker=current_speaker,
                        text=cleaned_text,
                        timestamp=timestamp
                    ))
                
                # Start new utterance
                current_speaker = detected_speaker
                current_text_parts = [text_content] if text_content else []
            
            else:
                # Continue previous speaker's utterance
                if current_speaker and line:
                    current_text_parts.append(line)
        
        # Don't forget the last utterance
        if current_speaker and current_text_parts:
            combined_text = ' '.join(current_text_parts)
            timestamp, cleaned_text = self.extract_timestamp(combined_text)
            
            utterances.append(RawUtterance(
                speaker=current_speaker,
                text=cleaned_text,
                timestamp=timestamp
            ))
        
        return utterances
    
    def create_structured_utterances(self, raw_utterances: List[RawUtterance]) -> List[Dict]:
        """Convert raw utterances to schema-compliant JSON objects."""
        structured = []
        
        for i, utterance in enumerate(raw_utterances):
            if not utterance.text.strip():
                continue
                
            structured_utterance = {
                "utterance_id": f"UTT_{i+1:04d}",
                "speaker": utterance.speaker,
                "turn_number": i + 1,
                "text": utterance.text.strip(),
                "word_count": len(utterance.text.split()),
                "features": {
                    "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
                    "speech_act": "assertion",  # Default, will be refined
                    "pronouns": {"first_person": 0, "second_person": 0, "third_person": 0},
                    "linguistic_markers": {
                        "hedging_words": 0,
                        "intensifiers": 0,
                        "modal_verbs": 0,
                        "imperatives": 0
                    },
                    "relational_indicators": {
                        "consent_language": False,
                        "boundary_language": False,
                        "accountability_language": False,
                        "support_language": False,
                        "power_assertion": False
                    }
                }
            }
            
            if utterance.timestamp:
                structured_utterance["timestamp"] = utterance.timestamp
                
            structured.append(structured_utterance)
        
        return structured

def main():
    """Process the conversation data from your provided text."""
    
    # Your conversation data (reformatted for parsing)
    conversation_text = """
Person A: Nothing?

Person B: That was a stupid attempt to compensate for not having taken you on a date before anything physical happened between us. 

I care about you so much, and I have such a high opinion of you, that I did not believe that you ever should have been in a mental hospital. 
Because you were highly articulate, and an amazing person I felt like I was getting to know you,, at a time that you had been confined when you should never have been in the hospital.  

The conditions that you have - ASD, ADHD and complex PTSD - don't justify forcing someone to stay in a mental hospital. Complex PTSD is varied in presentation. I believed that your dad had manipulated you into checking in to the hospital, and that the reason that you were so frustrated at being there was because you shouldn't have been in there at all. 

I didn't understand the reason for men being blocked from seeing you. 

I felt like I had been respectful when talking to you, and that things had moved quickly, but I felt like that was due to the strength of our connection, not your vulnerability. 
You made me feel special. 
You were kind to me and helped me to accept that I had been in an abusive relationship. 

I felt like I understood you, but I was only starting to learn who you are. It was only parts of you that I understood. 

It was only when you left and Jack moved in that I understood how vulnerable you are. 

Because I considered that I was respectful and asking how you felt and what you wanted, I was unable to see your feelings of obligation that came from past experiences. 

What I couldn't see in myself, I was able to in someone else. 

You are right. 
I do have some very big blind spots. 

I am sorry.

So at the same time that you have been doing so well, engaging with support services and moving forward, you have had to rely on a clueless dickhead to get you there.

You did tell me that you were vulnerable while you were in hospital too. 
I made the assumption that because I am respectful and value consent and autonomy, you would not feel any obligation to me. I was wrong to assume. 

That failed to take into account the abuse that you had experienced and the survival strategies that developed as a result.

I also failed to consider the perceived power differential between us. 
I had never heard the term before. 
I don't consider myself above you, or anyone. 
But that doesn't change how it felt to you. 
Here was someone who was confident that they could solve most of your problems. And then they were attracted to you, and had spoken about wanting to pursue a relationship at some point. So when there was an indication of flirting, you would have thought, oh well, if it gets me what I need. 

Person A: "That was a stupid attempt to compensate for not having taken you on a date before anything physical happened between us."

Was it?

Do you still deny you said "I tried to stop it" twice in reference to our having sex? You said it when I invited you to consider whether you had any obligation to be transparent with my friend about your intent toward me. You said your actions were ethical. You didn't deceive my friend. You put down his shortcomings. You promoted yourself stating you wish we tried to stop it and you wish we had waited. 

…

Are you accusing me of manipulating you?

Person B: No, I am not. 
This was my failure. 

Person A: "You would have thought oh well if it gets me what I need"

It is an accusation of manipulation and carelessness.

Person B: It is not. 
This is on me. 

When you have had to deal with abuse in other situations, and I failed to consider how my flirting was likely to be taken by you, and you still had the same need to move forward. 

I have continued getting to know you and learn more about you. I care about you. I value you and your happiness. 
I am sorry that I didn't understand. I will do everything that I can to do better in the future and to atone for how I made you feel. 

I do make mistakes. 
I am a long way from perfect. 
But I do my best to learn from them and not repeat them. 

You did not manipulate me. 
You were not careless 
You were kind, and gentle (emotionally) and caring. 

I am still here. 
I am ashamed. 
But I am still here. 
I will apologise to Garrett. 

Person A: Why did you state "you would have thought oh well it gets me what I need" if it's on you?

I paid you what you earned. As I said previously, you would start touching me with no indicator from me. You would not stop. You would sit and place your hands in my shirt and stay that way rubbing my nipples harder and harder. I did not respond. I sat at my laptop entirely focussed my laptop. Not for a short time. I did not feel or notice it until it got uncomfortable.   

I could not understand how you did not read the room. It confused me. I did not know what to say or do it was so odd. 

You helped me We all know reciprocity is a cultural norm.

Person B: Because it indicated your feeling of obligation. 

I did not 'earn' the gift of yourself. 
I am sorry Tia. 
I am not normal. 
I will learn from this and be a better man. 

You are worth more than my support and assistance with these things. 

There will always be a positive outcome when you tell me about something like this. 
You give me the opportunity to change and grow. 
Thank you 
"""
    
    parser = ConversationParser()
    
    # Create output directories
    output_dir = Path("/Users/tiaastor/relational_discourse_project")
    data_dir = output_dir / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse the conversation
    raw_utterances = parser.parse_conversation_text(conversation_text)
    structured_utterances = parser.create_structured_utterances(raw_utterances)
    
    # Save Person A utterances
    person_a_utterances = [u for u in structured_utterances if u["speaker"] == "PERSON_A"]
    with open(processed_dir / "person_a_utterances.json", 'w') as f:
        json.dump(person_a_utterances, f, indent=2)
    
    # Save Person B utterances  
    person_b_utterances = [u for u in structured_utterances if u["speaker"] == "PERSON_B"]
    with open(processed_dir / "person_b_utterances.json", 'w') as f:
        json.dump(person_b_utterances, f, indent=2)
    
    # Save complete conversation
    with open(processed_dir / "complete_conversation.json", 'w') as f:
        json.dump(structured_utterances, f, indent=2)
    
    # Create summary report
    summary = {
        "total_utterances": len(structured_utterances),
        "person_a_turns": len(person_a_utterances),
        "person_b_turns": len(person_b_utterances),
        "person_a_word_count": sum(u["word_count"] for u in person_a_utterances),
        "person_b_word_count": sum(u["word_count"] for u in person_b_utterances),
        "conversation_balance": len(person_b_utterances) / len(person_a_utterances) if person_a_utterances else 0
    }
    
    with open(processed_dir / "conversation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Parsed conversation into {len(structured_utterances)} utterances")
    print(f"📊 Person A: {len(person_a_utterances)} turns, {summary['person_a_word_count']} words")
    print(f"📊 Person B: {len(person_b_utterances)} turns, {summary['person_b_word_count']} words")
    print(f"⚖️  Turn ratio (B:A): {summary['conversation_balance']:.2f}")

if __name__ == "__main__":
    main()