# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public BytePlus TTS parameter types and protocol limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

TTSResourceID = Literal[
    "seed-tts-1.0",
    "seed-tts-2.0",
    "seed-icl-1.0",
    "seed-icl-2.0",
]
# Kept for compatibility with the standard LiveKit ``model`` constructor argument.
TTSModel = TTSResourceID

TTSVoice = Literal[
    # TTS 2.0
    "zh_female_vv_uranus_bigtts",  # Female | A youthful and vibrant female voice
    "zh_female_xiaohe_uranus_bigtts",  # Female | A gentle, soft-spoken, and slightly mature female voice
    "en_female_stokie_uranus_bigtts",  # Female | A trendy, casual, and expressive young female voice
    "en_female_dacey_uranus_bigtts",  # Female | A warm, empathetic, and highly engaging female voice
    "en_male_tim_uranus_bigtts",  # Male | A clear, versatile, and friendly mid-range male voice
    "zh_male_m191_uranus_bigtts",  # Male | A steady, clear, and versatile mid-range male voice
    "zh_male_taocheng_uranus_bigtts",  # Male | A dynamic, spirited, and energetic young male voice
    "zh_male_sophie_uranus_bigtts",  # Female | A smooth, modern, and soft-spoken young male voice
    "zh_female_yingyujiaoxue_uranus_bigtts",  # Female | A clear, authoritative yet encouraging female voice
    "zh_male_dayi_uranus_bigtts",  # Male | A mature, resonant, and slightly dramatic male voice
    "zh_female_mizai_uranus_bigtts",  # Female | A youthful, playful, and slightly quirky voice with a kid vibe
    "zh_female_jitangnv_uranus_bigtts",  # Female | A mature, soothing, and deeply emotional female voice
    "zh_female_meilinvyou_uranus_bigtts",  # Female | A sweet, intimate, and affectionate young female voice
    "zh_female_liuchangnv_uranus_bigtts",  # Female | A clear, steady, and highly articulate female voice
    "zh_male_ruyayichen_uranus_bigtts",  # Male | A refined, gentle, and elegant young male voice
    "zh_female_cancan_uranus_bigtts",  # Female | A young, vivid and energetic female voice
    "zh_female_tianmeixiaoyuan_uranus_bigtts",  # Female | A fresh, innocent, and youthful female voice
    "zh_female_tianmeitaozi_uranus_bigtts",  # Female | A soft, bright, and incredibly sweet young female voice
    "zh_female_shuangkuaisisi_uranus_bigtts",  # Female | A crisp, fast-paced, and confident young female voice
    "zh_female_peiqi_uranus_bigtts",  # Female | A childlike, innocent, and deeply endearing little girl's voice
    "zh_female_xiaoxue_uranus_bigtts",  # Female | A pure, steady, and crystal-clear young female voice
    "zh_female_yuanqi_uranus_bigtts",  # Female | An energetic, cheerful, and optimistic young female voice
    "zh_female_kefunvsheng_uranus_bigtts",  # Female | A polished, professional, and courteous female voice
    "zh_male_shaonianzixin_uranus_bigtts",  # Male | A bright, confident, and energetic teenage male voice
    "zh_female_linjianvhai_uranus_bigtts",  # Female | A warm, friendly, and approachable young female voice
    "zh_female_kiwi_uranus_bigtts",  # Female | A bright, cheerful, and modern young female voice
    "zh_female_sajiaoxuemei_uranus_bigtts",  # Female | A very young, sweet, and playful female voice
    "de_male_seven_uranus_bigtts",  # Male | A steady, clear, and confident male voice
    "jp_female_minimi_uranus_bigtts",  # Female | A high-pitched, sweet, and "kawaii" young female voice
    "fr_male_usseau_uranus_bigtts",  # Male | A sophisticated, crisp, and articulate male voice
    "es_male_felipe_uranus_bigtts",  # Male | An energetic, upbeat, and charismatic young male voice
    "id_male_han_uranus_bigtts",  # Male | A modern, smooth, and friendly young adult male voice
    "pt_male_martins_uranus_bigtts",  # Male | A charismatic, warm, and expressive male voice
    "it_male_enzo_uranus_bigtts",  # Male | An authentic, charismatic, and warm Italian male voice
    "kr_male_shane_uranus_bigtts",  # Male | A polished, modern, and smooth Korean male voice
    "zh_male_liufei_uranus_bigtts",  # Male | A clear and energetic voice
    "zh_female_qingxinnvsheng_uranus_bigtts",  # Female | A fresh and clear female voice
    "zh_male_sunwukong_uranus_bigtts",  # Male | A monkey king voice
    "en_male_alberto_uranus_bigtts",  # Male | A gentle, approachable man with a low, soothing tone
    "en_male_alex_uranus_bigtts",  # Male | A young man who is objective and composed, with a warm, clear voice
    "en_female_allison_uranus_bigtts",  # Female | An upbeat, enthusiastic college-aged woman, full of energy and warmth
    "en_male_bill_jones_corey_uranus_bigtts",  # Male | A steady, self-assured male professional with poise and composure
    "en_male_brad_pitt_p1_uranus_bigtts",  # Male | A laid-back man with a low, husky voice and a relaxed, easygoing manner
    "en_female_brittney_uranus_bigtts",  # Female | A warm, intelligent older sister with a tender heart
    "en_female_brittney_pimintel_uranus_bigtts",  # Female | A bright, spirited young girl bursting with energy
    "en_male_bruce_uranus_bigtts",  # Male | A composed, level-headed gentleman with rational restraint
    "en_male_chandler_p1_uranus_bigtts",  # Male | A theatrical man with exaggerated, dramatic intonation and rich expressiveness
    "en_male_cowboy_john_b_uranus_bigtts",  # Male | An energetic, flamboyant uncle with a Southern American accent
    "en_male_david_uranus_bigtts",  # Male | A middle-aged man with a deep, weighty voice, an unhurried pace, and natural pauses
    "en_male_diyuwenrounan_uranus_bigtts",  # Male | A refined, gentle man who is sincere and friendly, with a relaxed, easygoing way of speaking
    "en_male_godfather_uranus_bigtts",  # Male | A mature man with sincere emotion who tells his story in a gentle, unhurried way
    "en_male_gollum_uranus_bigtts",  # Male | A wacky, over-the-top male voice, skilled at creating playful characters
    "en_male_hades_uranus_bigtts",  # Male | A free-spirited mature man with a relaxed, easygoing way of speaking
    "en_female_hayley_uranus_bigtts",  # Female | A lively female voice with strong emotional tension, skilled at storytelling
    "en_male_jamie_uranus_bigtts",  # Male | A hearty male voice - sincere and candid, witty and full of energy
    "en_female_jane_uranus_bigtts",  # Female | An energetic young girl with outwardly vivid, expressive emotion
    "en_female_jenny_uranus_bigtts",  # Female | A naturally cheerful, smiling, warm and talkative personality
    "en_male_jidongchuanjiaoshi_uranus_bigtts",  # Male | An immersive, passionate performance with high-spirited, fervent intonation
    "en_male_jimmy_uranus_bigtts",  # Male | A sunny young man, vivid and engaging, skilled at sharing stories
    "en_female_joanne_uranus_bigtts",  # Female | A crisp, lively American young female voice with a light tone and a relaxed, natural conversational feel
    "en_male_joker_uranus_bigtts",  # Male | An American middle-aged male voice with a slow pace and a warm, magnetic tone
    "en_male_josh_uranus_bigtts",  # Male | A clear, bright young man - cheerful, energetic, and easygoing
    "en_male_josh_coery_uranus_bigtts",  # Male | A businesslike young man with a deep, magnetic voice and a dignified, steady tone
    "en_male_kevin_uranus_bigtts",  # Male | An American middle-aged voice - warm and magnetic, with vivid, fluent delivery
    "en_male_knightley_uranus_bigtts",  # Male | A deep, magnetic American middle-aged male voice with a steady, dignified manner
    "en_female_lana_del_rey_kelley_d_p1_uranus_bigtts",  # Female | An American young female voice - soft and slightly husky
    "en_female_lana_del_rey_parky_s_p1_uranus_bigtts",  # Female | A gentle, soft female voice with a warm, natural tone that is soothing to the ear
    "en_male_marcus_uranus_bigtts",  # Male | A mature American male voice - mellow and deep, skilled at gentle, unhurried storytelling
    "en_female_mel_uranus_bigtts",  # Female | A lively, dynamic American female voice with vivid ups and downs, skilled at performing animated dialogue
    "en_male_michael_uranus_bigtts",  # Male | A laid-back man with a deep, magnetic voice and a relaxed, easygoing manner
    "en_male_michael_kevin_uranus_bigtts",  # Male | A professional male narrator - gentle and bright, with highly persuasive delivery
    "en_female_myra_uranus_bigtts",  # Female | A sweet, lively young-girl voice that tells stories in a gentle, moving way
    "en_female_myra_cmb_uranus_bigtts",  # Female | A crisp, lively American young female voice - full of enthusiasm and outstanding expressiveness
    "en_female_nadia_uranus_bigtts",  # Female | A young American female voice - clean and clear, with a gentle, relaxed way of speaking
    "en_female_natasha_uranus_bigtts",  # Female | A conversational female voice - bright and vivid, warm and approachable
    "en_female_rachel_p1_uranus_bigtts",  # Female | An outwardly expressive young female voice - bright and clear, with strong dramatic flair
    "en_male_ronald_uranus_bigtts",  # Male | A British gentleman with a deep, resonant, dignified voice
    "en_male_russell_uranus_bigtts",  # Male | An American boyfriend voice - warm and bright, sincere and friendly
    "en_female_scarlet_p1_uranus_bigtts",  # Female | A gentle, deeply affectionate older sister whose eyes always hold a spark of light
    "en_female_sharron_uranus_bigtts",  # Female | A young lady with a soft, slightly husky voice and a leisurely, easygoing tone
    "en_male_simba_p1_uranus_bigtts",  # Male | A bright yet husky American young male voice with extremely strong dramatic expressiveness
    "en_female_skye_uranus_bigtts",  # Female | A clear, candid older sister who speaks sincerely and from the heart
    "en_male_tom_hiddleston_p1_uranus_bigtts",  # Male | A deep, reserved uncle whose reciting style is full of narrative feeling.
    "en_male_valentino_uranus_bigtts",  # Male | A sunny, cheerful young man, full of warmth and contagious energy.
    "en_male_valentino_corey_uranus_bigtts",  # Male | A mature, dignified uncle with American pronunciation and a steady, powerful professional narration style
    "en_female_wenrouzhishijieshuonv_uranus_bigtts",  # Female | A gentle, fun older sister with American pronunciation who shares interesting knowledge in a relaxed way.
    "en_female_xinwenjieshuonv_uranus_bigtts",  # Female | An enthusiastic, outgoing female college student with American pronunciation, full of emotion and expressiveness.
    "en_male_yangguangjieshuonan_uranus_bigtts",  # Male | A witty, humorous uncle with American pronunciation and a vivid, expressive narrative style.
    "en_female_zendaya_p1_uranus_bigtts",  # Female | An easygoing, approachable older sister - relaxed and unpretentious, yet full of energy.
    "ja_female_bv024_uranus_bigtts",  # Female | A gentle, soft female college student - friendly and warm, with an easygoing, unreserved tone.
    "ja_female_bv520_uranus_bigtts",  # Female | An energetic young woman with an anime-and-advertising dubbing style and fervent emotion.
    "ja_female_bv521_uranus_bigtts",  # Female | A sweet, lively young woman with a Japanese-style girl voice and strong performance appeal.
    "ja_female_bv522_uranus_bigtts",  # Female | A professional, dignified young woman with a standard Japanese broadcasting tone - calm and objective.
    "ja_female_bv523_uranus_bigtts",  # Female | An innocent, carefree little girl with a lively tone, full of childlike charm.
    "ja_male_bv524_uranus_bigtts",  # Male | A capable, professional young man with a Japanese broadcast-narration style and a calm, restrained tone.
    "ja_female_minimi_uranus_bigtts",  # Female | A vibrant, hearty young woman with an outgoing personality and outwardly full emotion.
    "ja_female_shirou_uranus_bigtts",  # Female | A spirited, straightforward young woman with a distinct anime-and-game dubbing style.
    "de_female_bv081_uranus_bigtts",  # Female | A young female voice that is objective, composed, and professional
    "de_male_sven_uranus_bigtts",  # Male | A middle-aged man with a deep, magnetic, slightly raspy voice
    "es_female_bv084_uranus_bigtts",  # Female | A rational, capable woman - calm and composed, with clear, well-organized narration.
    "es_male_dani_uranus_bigtts",  # Male | An enthusiastic, talkative uncle with European Spanish pronunciation and a fluent, charming narrative style
    "es_male_guillem_uranus_bigtts",  # Male | An easygoing, cheerful young man - relaxed and natural, and highly approachable.
    "es_female_ht_mx_f6_uranus_bigtts",  # Female | A lively, enthusiastic girl-next-door with Latin American Spanish pronunciation, full of energy.
    "mx_female_bv065_uranus_bigtts",  # Female | A calm, objective, capable young woman who is efficient and well-organized.
    "mx_male_bv165dialogue_uranus_bigtts",  # Male | A charming young man with a Spanish film-and-drama dialogue style - magnetic voice and vivid delivery.
    "mx_male_bv165narrator_uranus_bigtts",  # Male | A steady, mature workplace heartthrob with a deep, magnetic voice and a calm, professional tone.
    "mx_female_bv166dialogue_uranus_bigtts",  # Female | A playful, cheerful, lovely young woman with vivid, lively emotional expression.
    "mx_female_bv166emotion_uranus_bigtts",  # Female | A young woman with intense emotion and full dramatic tension - rich in expressiveness.
    "mx_female_bv166narrator_uranus_bigtts",  # Female | A young woman with a distinct style - full of emotion and skilled at vivid storytelling.
    "mx_male_felipe_uranus_bigtts",  # Male | An enthusiastic, sharp-witted cheerful young man, skilled at building a suspenseful atmosphere.
    "mx_male_ht_mx_m012_uranus_bigtts",  # Male | An objective, restrained young man with a distinctly professional narration style - composed and unflappable.
    "mx_female_leslie_uranus_bigtts",  # Female | An approachable, gentle young woman with a soft, natural pace.
    "mx_male_marcelo_uranus_bigtts",  # Male | A gentle, refined young man - steady and dignified.
    "fr_female_fr_bv078_uranus_bigtts",  # Female | A steady, professional auntie with standard French pronunciation - objective and even-tempered.
    "fr_female_fr_f47_uranus_bigtts",  # Female | A professional, capable older sister with fluent French pronunciation and a calm, objective narration style
    "fr_male_fr_m29_uranus_bigtts",  # Male | A mature, steady uncle with a resonant French voice and a professional, dignified narration style
    "id_male_bv160_uranus_bigtts",  # Male | An impassioned young man - angry and defiant, with intensely dramatic delivery.
    "id_male_bv160dialogue_uranus_bigtts",  # Male | A young man with outstanding expressiveness - full of emotion and a dramatic performance style.
    "id_male_bv160narration_uranus_bigtts",  # Male | A rational, steady young man with a calm, restrained tone, skilled at narrative storytelling.
    "id_female_bv161_uranus_bigtts",  # Female | A gentle, even-tempered young woman with a bright, soft voice and a strong narrative reading feel.
    "id_female_bv161dialogue_uranus_bigtts",  # Female | A young woman with shifting emotions - at ease in everyday talk or heated argument, bringing full film-and-TV dialogue atmosphere.
    "id_female_bv161narration_uranus_bigtts",  # Female | A sweet, lively female lead with a crisp voice and emotionally rich narrative and explanatory delivery.
    "id_female_bv164_uranus_bigtts",  # Female | A blend of diverse voices - natural, vivid multi-person dialogue with a rich storytelling atmosphere.
    "id_male_bv164dialogue_uranus_bigtts",  # Male | A gentle, refined young man with a soft voice and a distinctly dramatic performance style.
    "id_male_bv164narration_uranus_bigtts",  # Male | A professional, steady mature uncle - calm and objective in his approach.
    "id_female_f20_uranus_bigtts",  # Female | An energetic, lively young woman with a graceful, elegant air.
    "id_male_m08_uranus_bigtts",  # Male | A rational, steady young man with a calm tone and a professional narration style.
    "id_female_phulia_uranus_bigtts",  # Female | A lively, cheerful young woman - full of emotion and a performance style brimming with tension.
    "pt_male_bv172_uranus_bigtts",  # Male | A two-person dialogue featuring a regular young male voice and a deep, husky middle-aged male voice, with a striking contrast between fast and slow pacing.
    "pt_male_bv172dialogue_uranus_bigtts",  # Male | A resonant, husky middle-aged uncle with a serious tone and a strong film-and-TV dialogue feel.
    "pt_male_bv172emotion_uranus_bigtts",  # Male | A middle-aged male voice with intense emotion, exaggerated intonation, and standout dramatic expressiveness.
    "pt_male_bv172narrator_uranus_bigtts",  # Male | A rational, calm male narrator - rigorous and professional.
    "pt_female_bv173_uranus_bigtts",  # Female | A professional, composed female narrator - clear-minded, sharp, and capable.
    "pt_female_bv173dialogue_uranus_bigtts",  # Female | An elegant, mature idol-drama female lead - delicate in thought, with a lively, captivating air.
    "pt_female_bv173emotion_uranus_bigtts",  # Female | A performance sprite who loves the dramatic stage - enthusiastic, full of emotion, and brimming with expressive tension.
    "pt_female_bv173narrator_uranus_bigtts",  # Female | A sharp, capable professional female narrator - calm and rational, with a steady, commanding presence.
    "pt_female_bv530_uranus_bigtts",  # Female | A gentle, soft, approachable young woman - lively and natural, skilled in Brazilian Portuguese.
    "pt_male_bv531_uranus_bigtts",  # Male | A rational, objective middle-aged male voice - steady and dependable.
    "pt_female_mari_uranus_bigtts",  # Female | A cheerful young woman with a hearty, commanding voice and a talkative, gracious manner.
    "pt_male_rael_uranus_bigtts",  # Male | A fresh, crisp young man - energetic and sunny, skilled in Brazilian Portuguese.
    "ar_female_dina_uranus_bigtts",  # Female | A warm and lively Egyptian woman, deeply versed in local culture
    "ar_female_fatma_uranus_bigtts",  # Female | A young woman with a gentle, tender voice, often heard in soft solo monologues
    "ar_male_youssef_uranus_bigtts",  # Male | A middle-aged man with a calm, easygoing tone, speaking in an intimate, conversational manner
    "tl_female_annika_uranus_bigtts",  # Female | An approachable young woman with a gentle tone and an authentic everyday-conversation quality.
    "tl_male_ed_uranus_bigtts",  # Male | An approachable, easygoing middle-aged male voice with relaxed, down-to-earth delivery.
    "tl_female_hervie_uranus_bigtts",  # Female | An entertainment-news female anchor - professional, confident, and engaging.
    "ko_male_bv545_uranus_bigtts",  # Male | A hearty young man with a Korean-style, true-to-life performance - full of emotion and natural expression.
    "ko_female_bv546_uranus_bigtts",  # Female | A candid, lively young woman with an anime-and-drama dubbing style and abundant emotion.
    "ko_male_m03_uranus_bigtts",  # Male | A standard Korean male narrator - highly professional, with a magnetic voice and a steady tone.
    "ko_male_shane_uranus_bigtts",  # Male | A steady, refined middle-aged uncle with a highly persuasive tone.
    "ms_male_ham_uranus_bigtts",  # Male | A steady, easygoing uncle with a Malay everyday-conversation style, an even tone, and a talent for analysis and explanation.
    "ms_male_naim_uranus_bigtts",  # Male | A gentle, refined middle-aged uncle - calm and reserved, rigorous and dependable.
    "ru_female_af07_uranus_bigtts",  # Female | A gentle, approachable young woman - understanding and graceful.
    "ru_female_irinae_uranus_bigtts",  # Female | An enthusiastic Russian young woman with outwardly expressive, rich emotion.
    "ru_male_pavel_uranus_bigtts",  # Male | A middle-aged male voice with a naturally true-to-life narrative tone - comfortable and natural to listen to.
    "ru_female_sophie_uranus_bigtts",  # Female | A young female voice with a Russian accent - full of energy and warmth, with a strong sense of connection.
    "ru_male_vlad_uranus_bigtts",  # Male | A soft-spoken young man - gentle and reserved, calm and even-tempered.
    "th_female_bv568_angry_uranus_bigtts",  # Female | A domineering female lead with shifting emotions - highly expressive and outwardly emotional.
    "th_female_bv568_fear_uranus_bigtts",  # Female | An anxious, fearful innocent young woman - emotionally fragile and sensitive.
    "th_female_bv568_happy_uranus_bigtts",  # Female | A young woman with full emotion and the quality of both dramatic and animation dubbing - rich in expressiveness.
    "th_female_bv568_hate_uranus_bigtts",  # Female | A female lead with strong dramatic tension - full, expressive emotion with a distinct film-and-TV dialogue quality.
    "th_female_bv568_neutral_uranus_bigtts",  # Female | A calm, even-tempered young woman with balanced emotion - calm and neutral in her approach.
    "th_female_bv568_sad_uranus_bigtts",  # Female | A gentle, melancholic young woman with a strong sense of narrative.
    "th_female_bv568_suprise_uranus_bigtts",  # Female | A lively, vivid-minded young woman with a proactive personality and a naturally playful, teasing air.
    "vi_female_hong_uranus_bigtts",  # Female | An out-of-town young woman with a Vietnamese accent - straightforward, with frank emotional expression.
    "vi_female_ling_uranus_bigtts",  # Female | A tender, kind young woman - earnest and dependable in her work.
    "vi_female_linh_uranus_bigtts",  # Female | A straightforward, candid young woman full of energy - crisp and decisive in expression.
    "vi_female_partner_uranus_bigtts",  # Female | A young woman with intense, full emotion - brimming with youthful vitality.
    "vi_female_ruan_uranus_bigtts",  # Female | A steady, well-measured young woman - clear-minded, dignified, and poised.
    "vi_female_wu_uranus_bigtts",  # Female | A straightforward, outgoing young woman with a steady, rational mindset.
    "vi_male_wumg_uranus_bigtts",  # Male | A modest, patient young man - rigorous and meticulous.
    # TTS 1.0
    "en_female_candice_emo_v2_mars_bigtts",  # Female | Warm
    "en_female_skye_emo_v2_mars_bigtts",  # Female | Vivid
    "en_male_glen_emo_v2_mars_bigtts",  # Male | Clear
    "en_male_sylus_emo_v2_mars_bigtts",  # Male | Deep
    "en_male_corey_emo_v2_mars_bigtts",  # Male | Clear
    "en_female_nadia_tips_emo_v2_mars_bigtts",  # Female | Sweet
    "en_female_lauren_moon_bigtts",  # Female | Vivid
    "en_male_campaign_jamal_moon_bigtts",  # Male | Clear
    "en_male_chris_moon_bigtts",  # Male | Deep
    "en_female_product_darcie_moon_bigtts",  # Female | Warm
    "en_female_emotional_moon_bigtts",  # Female | Sweet
    "en_female_nara_moon_bigtts",  # Female | Deep
    "en_male_bruce_moon_bigtts",  # Male | Deep
    "en_male_michael_moon_bigtts",  # Male | Elegant
    "ICL_en_male_cc_sha_v1_tob",  # Male | Deep
    "zh_male_M100_conversation_wvae_bigtts",  # Male | Elegant
    "zh_female_sophie_conversation_wvae_bigtts",  # Female | Warm
    "en_female_dacey_conversation_wvae_bigtts",  # Female | Clear
    "en_male_charlie_conversation_wvae_bigtts",  # Male | Deep
    "en_female_sarah_new_conversation_wvae_bigtts",  # Female | Elegant
    "ICL_en_male_michael_tob",  # Male | Soft
    "ICL_en_female_cc_cm_v1_tob",  # Female | Vivid
    "ICL_en_male_oogie2_tob",  # Male | Deep
    "ICL_en_male_frosty1_tob",  # Male | Deep
    "ICL_en_male_grinch2_tob",  # Male | Deep
    "ICL_en_male_zayne_tob",  # Male | Clear
    "ICL_en_male_cc_jigsaw_tob",  # Male | Deep
    "ICL_en_male_cc_chucky_tob",  # Male | Clear
    "ICL_en_male_cc_penny_v1_tob",  # Male | Deep
    "ICL_en_male_kevin2_tob",  # Male | Clear
    "ICL_en_male_xavier1_v1_tob",  # Male | Mature
    "ICL_en_male_cc_dracula_v1_tob",  # Male | Mature
    "en_female_daisy_moon_bigtts",  # Female | Clear
    "en_male_dave_moon_bigtts",  # Male | Deep
    "en_male_hades_moon_bigtts",  # Male | Deep
    "en_female_onez_moon_bigtts",  # Female | Soft
    "en_female_emily_mars_bigtts",  # Female | Soft
    "zh_male_xudong_conversation_wvae_bigtts",  # Male | Deep
    "ICL_en_male_cc_alastor_tob",  # Male | Vivid
    "ICL_en_male_aussie_v1_tob",  # Male | Warm
    "zh_female_shuangkuaisisi_emo_v2_mars_bigtts",  # Female | Vivid
    "zh_female_shaoergushi_mars_bigtts",  # Female | Vivid
    "zh_male_silang_mars_bigtts",  # Male | Deep
    "zh_male_jieshuonansheng_mars_bigtts",  # Male | Clear
    "zh_female_jitangmeimei_mars_bigtts",  # Female | Soft
    "zh_female_tiexinnvsheng_mars_bigtts",  # Female | Warm
    "zh_female_qiaopinvsheng_mars_bigtts",  # Female | Vivid
    "zh_female_mengyatou_mars_bigtts",  # Female | Vivid
    "zh_female_cancan_mars_bigtts",  # Female | Clear
    "zh_female_qingxinnvsheng_mars_bigtts",  # Female | Clear
    "zh_female_linjia_mars_bigtts",  # Female | Vivid
    "zh_male_wennuanahu_moon_bigtts",  # Male | Warm
    "zh_male_shaonianzixin_moon_bigtts",  # Male | Clear
    "zh_female_shuangkuaisisi_moon_bigtts",  # Female | Vivid
    "en_female_anna_mars_bigtts",  # Female | Soft
    "en_male_adam_mars_bigtts",  # Male | Clear
    "en_female_sarah_mars_bigtts",  # Female | Soft
    "en_male_dryw_mars_bigtts",  # Male | Deep
    "en_male_smith_mars_bigtts",  # Male | Deep
    "zh_female_tianxinxiaomei_emo_v2_mars_bigtts",  # Female | Sweet
    "zh_female_gaolengyujie_emo_v2_mars_bigtts",  # Female | Mature
    "zh_male_aojiaobazong_emo_v2_mars_bigtts",  # Male | Deep
    "zh_male_guangzhoudege_emo_mars_bigtts",  # Male | Elegant
    "zh_male_jingqiangkanye_emo_mars_bigtts",  # Male | Deep
    "zh_female_linjuayi_emo_v2_mars_bigtts",  # Female | Soft
    "zh_female_jiaochuan_mars_bigtts",  # Female | Sweet
    "zh_female_flattery_mars_bigtts",  # Female | Vivid
    "ICL_zh_female_chunzhenshaonv_e588402fb8ad_tob",  # Female | Mature
    "ICL_zh_female_ganli_v1_tob",  # Female | Sexy
    "ICL_zh_female_xiangliangya_v1_tob",  # Female | Sexy
    "ICL_zh_male_guaogongzi_v1_tob",  # Male | Clear
    "ICL_zh_female_bingjiao3_tob",  # Female | Charming
    "ICL_zh_male_cujingnanyou_tob",  # Male | Clear
    "ICL_zh_male_shuanglangshaonian_tob",  # Male | Deep
    "ICL_zh_male_sajiaonanyou_tob",  # Male | Clear
    "ICL_zh_male_wenrounanyou_tob",  # Male | Deep
    "ICL_zh_male_wenshunshaonian_tob",  # Male | Warm
    "ICL_zh_male_tiancaitongzhuo_tob",  # Male | Clear
    "ICL_zh_male_aojiaojingying_tob",  # Male | Deep
    "ICL_zh_male_bingjiaoshaonian_tob",  # Male | Scheming
    "ICL_zh_male_jingyingqingnian_tob",  # Male | Mature
    "ICL_zh_male_fengfashaonian_tob",  # Male | Clear
    "ICL_zh_male_rexueshaonian_tob",  # Male | Clear
    "ICL_zh_male_lingyunqingnian_tob",  # Male | Deep
    "ICL_zh_male_ruyajunzi_tob",  # Male | Soft
    "ICL_zh_male_ruyazongcai_tob",  # Male | Deep
    "ICL_zh_male_cixingnansang_tob",  # Male | Elegant
    "ICL_zh_male_gaolengzongcai_tob",  # Male | Deep
    "ICL_zh_male_yuanqishaonian_tob",  # Male | Clear
    "zh_female_meilinvyou_moon_bigtts",  # Female | Sweet
    "zh_male_shenyeboke_moon_bigtts",  # Male | Deep
    "zh_female_sajiaonvyou_moon_bigtts",  # Female | Sweet
    "zh_female_yuanqinvyou_moon_bigtts",  # Female | Sweet
    "ICL_zh_female_bingruoshaonv_tob",  # Female | Sweet
    "ICL_zh_female_jiaoruoluoli_tob",  # Female | Sweet
    "ICL_zh_male_guzhibingjiao_tob",  # Male | Scheming
    "ICL_zh_male_sajiaonianren_tob",  # Male | Soft
    "ICL_zh_male_bingjiaobailian_tob",  # Male | Scheming
    "ICL_zh_female_bingjiaomengmei_tob",  # Female | Sexy
    "ICL_zh_female_aojiaonvyou_tob",  # Female | Scheming
    "ICL_zh_male_tiexinnanyou_tob",  # Male | Soft
    "ICL_zh_female_tiexinnvyou_tob",  # Female | Warm
    "ICL_zh_male_bingjiaodidi_tob",  # Male | Scheming
    "ICL_zh_male_aomanshaoye_tob",  # Male | Deep
    "ICL_zh_male_asmryexiu_tob",  # Male | Warm
    "ICL_zh_male_shenmifashi_tob",  # Male | Deep
    "zh_male_baqiqingshu_mars_bigtts",  # Male | Deep
    "zh_female_wenroushunv_mars_bigtts",  # Female | Soft
    "zh_female_gaolengyujie_moon_bigtts",  # Female | Clear
    "zh_female_linjianvhai_moon_bigtts",  # Female | Clear
    "zh_male_yuanboxiaoshu_moon_bigtts",  # Male | Deep
    "zh_male_yangguangqingnian_moon_bigtts",  # Male | Clear
    "zh_male_jingqiangkanye_moon_bigtts",  # Male | Fun
    "zh_male_guozhoudege_moon_bigtts",  # Male | Clear
    "zh_female_wanqudashu_moon_bigtts",  # Male | Fun
    "zh_female_daimengchuanmei_moon_bigtts",  # Female | Cute
    "zh_female_wanwanxiaohe_moon_bigtts",  # Female | Vivid
    "zh_male_zhoujielun_emo_v2_mars_bigtts",  # Male | Deep
    "zh_female_yueyunv_mars_bigtts",  # Female | Warm
    "multi_male_jingqiangkanye_moon_bigtts",  # Male | Fun
    "multi_female_shuangkuaisisi_moon_bigtts",  # Female | Vivid
    "multi_male_wanqudashu_moon_bigtts",  # Male | Fun
    "multi_female_sophie_conversation_wvae_bigtts",  # Female | Soft
    "multi_male_xudong_conversation_wvae_bigtts",  # Male | Deep
    "multi_female_maomao_conversation_wvae_bigtts",  # Female | Clear
    "multi_female_gaolengyujie_moon_bigtts",  # Female | Clear
    "multi_zh_male_youyoujunzi_moon_bigtts",  # Male | Clear
    "multi_male_M100_conversation_wvae_bigtts",  # Male | Deep
]

TTSEmotion = Literal[
    "affectionate",
    "angry",
    "ASMR",
    "authoritative",
    "chat",
    "coldness",
    "depressed",
    "excited",
    "fear",
    "happy",
    "hate",
    "neutral",
    "sad",
    "surprised",
    "warm",
]
TTSExplicitLanguage = Literal[
    "zh-cn",
    "en",
    "ja",
    "es-mx",
    "es",
    "id",
    "pt-br",
    "pt",
    "ko",
    "it",
    "de",
    "fr",
    "th",
    "vi",
    "ru",
    "fil",
    "ms",
    "ar",
    "pl",
    "tr",
    "sv",
]
TTSContextLanguage = Literal["id", "es", "pt"]
TTSExplicitDialect = Literal["dongbei", "shaanxi", "sichuan"]
TTSAudioFormat = Literal["pcm", "mp3", "ogg_opus"]
TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 44100, 48000]
TTSLatexParser = Literal["v2"]
TTSParenthesisFilterLength = Literal[0, 100]
TTSSpeakerModel = Literal["seed-tts-2.0-standard"]

SUPPORTED_AUDIO_FORMATS = frozenset(get_args(TTSAudioFormat))
SUPPORTED_SAMPLE_RATES = frozenset(get_args(TTSSampleRate))
SUPPORTED_EXPLICIT_LANGUAGES = frozenset(get_args(TTSExplicitLanguage))
SUPPORTED_CONTEXT_LANGUAGES = frozenset(get_args(TTSContextLanguage))
SUPPORTED_EXPLICIT_DIALECTS = frozenset(get_args(TTSExplicitDialect))
SUPPORTED_LATEX_PARSERS = frozenset(get_args(TTSLatexParser))
SUPPORTED_PARENTHESIS_FILTER_LENGTHS = frozenset(get_args(TTSParenthesisFilterLength))

SPEECH_RATE_RANGE = (-50, 100)
LOUDNESS_RATE_RANGE = (-50, 100)
PITCH_RANGE = (-12, 12)
EMOTION_SCALE_RANGE = (1, 5)
SILENCE_DURATION_RANGE_MS = (0, 30000)
UNSUPPORTED_CHAR_RATIO_RANGE = (0.0, 1.0)
MIN_CUSTOM_BIT_RATE = 1
MIN_DEFAULT_BIT_RATE = 64000

ICL_2_RESOURCE_IDS = frozenset({"seed-icl-2.0"})
TIMESTAMP_RESOURCE_IDS = frozenset({"seed-tts-1.0", "seed-icl-1.0"})
SUBTITLE_RESOURCE_IDS = frozenset({"seed-tts-2.0", *ICL_2_RESOURCE_IDS})
CONTEXT_TEXT_RESOURCE_IDS = SUBTITLE_RESOURCE_IDS
COMPRESSED_AUDIO_FORMATS = frozenset({"mp3", "ogg_opus"})


@dataclass(frozen=True, slots=True)
class AIGCMetadata:
    """Metadata embedded into BytePlus MP3 or OGG Opus output."""

    enable: bool = True
    content_producer: str | None = None
    produce_id: str | None = None
    content_propagator: str | None = None
    propagate_id: str | None = None

    def to_dict(self) -> dict[str, bool | str]:
        """Return the provider payload without unset optional fields."""
        result: dict[str, bool | str] = {"enable": self.enable}
        for name in (
            "content_producer",
            "produce_id",
            "content_propagator",
            "propagate_id",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class TTSUsage:
    """Billable character usage returned by the provider."""

    request_id: str
    text_words: int
